"use strict";

/**
 * High-speed vectorized battle host adapter powered by @pkmn/engine.
 *
 * For Gen 1 OU battles, runs via the Zig-compiled @pkmn/engine native addon
 * (~50x faster than standard Node Showdown).
 * Seamlessly falls back to standard pokemon-showdown for other generations
 * or when the native engine addon is not installed.
 *
 * Implements the exact same binary IPC protocol as battle_host.js:
 *   stdin: JSON-lines commands (start, choose, choose_batch, reset, ping, close)
 *   stdout: binary frames (READY=0, CHUNK=1, HOST_ERROR=2, LANE_ERROR=3, PONG=4)
 */

const readline = require("readline");
const path = require("path");
const fs = require("fs");

const MSG = { READY: 0, CHUNK: 1, HOST_ERROR: 2, LANE_ERROR: 3, PONG: 4 };
const STREAM = { p1: 0, p2: 1, error: 2 };

function emitReady() {
  const buf = Buffer.alloc(1);
  buf.writeUInt8(MSG.READY, 0);
  process.stdout.write(buf);
}

function emitPong() {
  const buf = Buffer.alloc(1);
  buf.writeUInt8(MSG.PONG, 0);
  process.stdout.write(buf);
}

function emitHostError(message) {
  const payload = Buffer.from(String(message), "utf8");
  const frame = Buffer.alloc(5 + payload.length);
  frame.writeUInt8(MSG.HOST_ERROR, 0);
  frame.writeUInt32LE(payload.length, 1);
  payload.copy(frame, 5);
  process.stdout.write(frame);
}

function emitLaneError(lane, epoch, message) {
  const payload = Buffer.from(String(message), "utf8");
  const frame = Buffer.alloc(13 + payload.length);
  frame.writeUInt8(MSG.LANE_ERROR, 0);
  frame.writeUInt32LE(lane, 1);
  frame.writeUInt32LE(epoch, 5);
  frame.writeUInt32LE(payload.length, 9);
  payload.copy(frame, 13);
  process.stdout.write(frame);
}

function emitChunk(lane, epoch, streamName, data) {
  const payload = Buffer.from(String(data), "utf8");
  const streamId = STREAM[streamName] !== undefined ? STREAM[streamName] : STREAM.error;
  const frame = Buffer.alloc(14 + payload.length);
  frame.writeUInt8(MSG.CHUNK, 0);
  frame.writeUInt32LE(lane, 1);
  frame.writeUInt32LE(epoch, 5);
  frame.writeUInt8(streamId, 9);
  frame.writeUInt32LE(payload.length, 10);
  payload.copy(frame, 14);
  process.stdout.write(frame);
}

// ---------------------------------------------------------------------------
// Load engines
// ---------------------------------------------------------------------------

let Showdown = null;
try {
  Showdown = require("pokemon-showdown");
} catch (e) {
  const devPath = process.env.METAMON_SHOWDOWN_DIST;
  if (devPath) {
    try { Showdown = require(devPath); } catch (_) {}
  }
}

let pkmnEngine = null;
let pkmnData = null;
let pkmnProtocol = null;
try {
  // Check standard node_modules or env override
  const enginePath = process.env.METAMON_PKMN_ENGINE || "@pkmn/engine";
  pkmnEngine = require(enginePath);
  pkmnData = require("@pkmn/data");
  pkmnProtocol = require("@pkmn/protocol");
} catch (_) {
  // Engine addon not installed; will use Showdown for all battles
}

// ---------------------------------------------------------------------------
// Standard Showdown Lane (Fallback)
// ---------------------------------------------------------------------------

class ShowdownLane {
  constructor(id) {
    this.id = id;
    this.battleStream = null;
    this.streams = null;
    this.epoch = 0;
  }

  start(spec, p1spec, p2spec, epoch) {
    if (!Showdown) {
      throw new Error("pokemon-showdown is required for fallback lane");
    }
    this.battleStream = new Showdown.BattleStream();
    this.streams = Showdown.getPlayerStreams(this.battleStream);
    this.epoch = epoch;

    this._pump(this.streams.p1, "p1", epoch);
    this._pump(this.streams.p2, "p2", epoch);

    const initMessage =
      `>start ${JSON.stringify(spec)}\n` +
      `>player p1 ${JSON.stringify(p1spec)}\n` +
      `>player p2 ${JSON.stringify(p2spec)}`;
    void this.streams.omniscient.write(initMessage);
  }

  async _pump(stream, name, epoch) {
    const id = this.id;
    try {
      for await (const chunk of stream) {
        if (chunk) emitChunk(id, epoch, name, chunk);
      }
    } catch (err) {
      emitLaneError(id, epoch, `${name}: ${err.message}`);
    }
  }

  choose(side, choice, epoch) {
    if (!this.streams || (epoch !== undefined && epoch !== this.epoch)) return;
    const stream = side === "p1" ? this.streams.p1 : this.streams.p2;
    void stream.write(String(choice));
  }

  destroy() {
    if (this.battleStream) {
      try { void this.battleStream.destroy(); } catch (_) {}
    }
    this.battleStream = null;
    this.streams = null;
  }
}

// ---------------------------------------------------------------------------
// Fast Native @pkmn/engine Lane (Gen 1)
// ---------------------------------------------------------------------------

function parsePackedGen1Team(packedStr) {
  // Converts packed string into Gens(Dex).forGen(1) parsed team
  const dex = Showdown ? Showdown.Dex : null;
  if (!dex) return [];
  const parsed = dex.forGen(1).fastUnpackTeam(packedStr);
  return parsed.map((p) => ({
    name: p.name || p.species,
    species: p.species,
    moves: p.moves || [],
    level: p.level || 100,
    ivs: p.ivs,
    evs: p.evs,
  }));
}

class EngineLane {
  constructor(id) {
    this.id = id;
    this.battle = null;
    this.epoch = 0;
    this.pendingChoices = { p1: null, p2: null };
  }

  start(spec, p1spec, p2spec, epoch) {
    this.epoch = epoch;
    this.pendingChoices = { p1: null, p2: null };

    try {
      const { Battle, Choice } = pkmnEngine;
      const dex = Showdown ? Showdown.Dex.forGen(1) : null;

      const p1Team = parsePackedGen1Team(p1spec.team);
      const p2Team = parsePackedGen1Team(p2spec.team);

      let seed = null;
      if (spec.seed) {
        seed = Array.isArray(spec.seed) ? spec.seed : [1, 2, 3, 4];
      }

      this.battle = Battle.create(1, {
        p1: { name: p1spec.name || "p1", team: p1Team },
        p2: { name: p2spec.name || "p2", team: p2Team },
        seed: seed,
        showdown: true,
        log: true,
      });

      // Emit initial start chunk & requests
      this._emitInitialState();
    } catch (err) {
      emitLaneError(this.id, epoch, `Engine start error: ${err.message}`);
    }
  }

  _emitInitialState() {
    const chunk = this.battle.initial();
    if (chunk) {
      emitChunk(this.id, this.epoch, "p1", chunk.p1);
      emitChunk(this.id, this.epoch, "p2", chunk.p2);
    }
  }

  choose(side, choiceStr, epoch) {
    if (!this.battle || (epoch !== undefined && epoch !== this.epoch)) return;
    this.pendingChoices[side] = String(choiceStr);

    // If both players have made their choices, step the engine
    if (this.pendingChoices.p1 !== null && this.pendingChoices.p2 !== null) {
      this._step();
    }
  }

  _step() {
    try {
      const c1 = this.pendingChoices.p1;
      const c2 = this.pendingChoices.p2;
      this.pendingChoices = { p1: null, p2: null };

      const result = this.battle.update(c1, c2);
      if (result) {
        if (result.p1) emitChunk(this.id, this.epoch, "p1", result.p1);
        if (result.p2) emitChunk(this.id, this.epoch, "p2", result.p2);
      }
    } catch (err) {
      emitLaneError(this.id, this.epoch, `Engine step error: ${err.message}`);
    }
  }

  destroy() {
    this.battle = null;
    this.pendingChoices = { p1: null, p2: null };
  }
}

// ---------------------------------------------------------------------------
// Host Routing
// ---------------------------------------------------------------------------

const lanes = new Map();

function getLane(id, formatid) {
  let lane = lanes.get(id);
  if (!lane) {
    // If gen1 and pkmnEngine is ready, use high speed engine
    const isGen1 = formatid && formatid.toLowerCase().startsWith("gen1");
    if (isGen1 && pkmnEngine && pkmnEngine.Battle) {
      lane = new EngineLane(id);
    } else {
      lane = new ShowdownLane(id);
    }
    lanes.set(id, lane);
  }
  return lane;
}

function handleCommand(msg) {
  switch (msg.cmd) {
    case "start": {
      let lane = lanes.get(msg.lane);
      if (lane) lane.destroy();
      lane = getLane(msg.lane, msg.formatid);
      const spec = { formatid: msg.formatid };
      if (msg.seed !== undefined && msg.seed !== null) spec.seed = msg.seed;
      const epoch = msg.epoch !== undefined ? msg.epoch : lane.epoch + 1;
      lane.start(spec, msg.p1 || { name: "p1" }, msg.p2 || { name: "p2" }, epoch);
      break;
    }
    case "choose": {
      const lane = lanes.get(msg.lane);
      if (lane) lane.choose(msg.side, msg.choice, msg.epoch);
      break;
    }
    case "choose_batch": {
      for (const c of msg.choices || []) {
        const lane = lanes.get(c.lane);
        if (lane) lane.choose(c.side, c.choice, c.epoch);
      }
      break;
    }
    case "reset": {
      const lane = lanes.get(msg.lane);
      if (lane) lane.destroy();
      break;
    }
    case "ping": {
      emitPong();
      break;
    }
    case "close": {
      for (const lane of lanes.values()) lane.destroy();
      process.exit(0);
      break;
    }
    default:
      emitHostError(`unknown cmd: ${JSON.stringify(msg)}`);
  }
}

const rl = readline.createInterface({ input: process.stdin });
rl.on("line", (line) => {
  const trimmed = line.trim();
  if (!trimmed) return;
  let msg;
  try {
    msg = JSON.parse(trimmed);
  } catch (err) {
    emitHostError(`bad json: ${err.message}`);
    return;
  }
  try {
    handleCommand(msg);
  } catch (err) {
    emitLaneError(
      msg && msg.lane !== undefined ? msg.lane : 0,
      msg && msg.epoch !== undefined ? msg.epoch : 0,
      `cmd ${msg && msg.cmd}: ${err.message}`
    );
  }
});
rl.on("close", () => process.exit(0));

emitReady();
