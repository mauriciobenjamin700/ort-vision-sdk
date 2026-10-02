import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

/**
 * Tests for the RGBA ⇄ RGB conversions every decoded frame goes through.
 *
 * Both run a whole pixel as one `Uint32` on little-endian platforms. These
 * tests compare that against a plain byte-by-byte reference on every byte of a
 * frame whose values cover all of 0..255 in every channel, so an endianness or
 * shift mistake cannot hide. They also pin the opacity verdict that decides
 * whether the decoded canvas may stand in for the RGB image.
 *
 * Node has no `ImageData`; a minimal one is installed for the duration.
 */

class FakeImageData {
  constructor(
    public readonly data: Uint8ClampedArray,
    public readonly width: number,
    public readonly height: number,
  ) {}
}

beforeEach(() => {
  vi.stubGlobal("ImageData", FakeImageData);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

const { imageDataToRGB, imageDataToRGBChecked, rgbToImageData } = await import(
  "../src/core/canvas.js"
);
const { RGBImage } = await import("../src/types.js");

/**
 * An RGBA frame where each channel walks 0..255 at a different stride.
 *
 * @param alpha Alpha for every pixel, or a function of the pixel index.
 */
function frame(alpha: number | ((pixel: number) => number)): ImageData {
  const width = 37;
  const height = 29;
  const data = new Uint8ClampedArray(width * height * 4);
  for (let p = 0; p < width * height; p++) {
    data[p * 4] = p % 256;
    data[p * 4 + 1] = (p * 7) % 256;
    data[p * 4 + 2] = (p * 13 + 5) % 256;
    data[p * 4 + 3] = typeof alpha === "number" ? alpha : alpha(p);
  }
  return new FakeImageData(data, width, height) as unknown as ImageData;
}

describe("imageDataToRGBChecked", () => {
  it("drops alpha byte for byte", () => {
    const source = frame(255);
    const { image } = imageDataToRGBChecked(source);

    const expected = new Uint8Array(source.width * source.height * 3);
    for (let p = 0; p < source.width * source.height; p++) {
      expected[p * 3] = source.data[p * 4] as number;
      expected[p * 3 + 1] = source.data[p * 4 + 1] as number;
      expected[p * 3 + 2] = source.data[p * 4 + 2] as number;
    }
    expect(image.data).toEqual(expected);
    expect(imageDataToRGB(source).data).toEqual(expected);
  });

  it("reports a fully opaque frame as opaque", () => {
    expect(imageDataToRGBChecked(frame(255)).opaque).toBe(true);
  });

  it("reports a single translucent pixel", () => {
    expect(imageDataToRGBChecked(frame((p) => (p === 500 ? 254 : 255))).opaque).toBe(false);
  });

  it("reports a fully transparent frame as not opaque", () => {
    expect(imageDataToRGBChecked(frame(0)).opaque).toBe(false);
  });
});

describe("rgbToImageData", () => {
  it("round-trips RGB and forces alpha to 255", () => {
    const { image } = imageDataToRGBChecked(frame(255));

    const back = rgbToImageData(image);

    expect(back.data).toEqual(frame(255).data);
  });

  it("handles an image whose byte length is not a multiple of four", () => {
    const image = new RGBImage(new Uint8Array([1, 2, 3]), 1, 1);

    expect(Array.from(rgbToImageData(image).data)).toEqual([1, 2, 3, 255]);
  });
});
