import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["test/**/*.test.ts"],
    // Node env is enough — tests cover pure functions and value types.
    // Canvas-dependent code (preprocess/image, io/image) is not unit-tested
    // here; it gets covered end-to-end via the example/ demo in a real browser.
    environment: "node",
  },
});
