/**
 * Common foundation for task-oriented vision SDK objects.
 *
 * The base only owns the {@link OrtSession} — label resolution lives in each
 * subclass because how `numClasses` is read from the model differs per task.
 */

import { OrtSession } from "../core/session.js";

export abstract class VisionTask {
  protected constructor(protected readonly _session: OrtSession) {}

  /** The underlying {@link OrtSession} used to run inference. */
  get session(): OrtSession {
    return this._session;
  }
}
