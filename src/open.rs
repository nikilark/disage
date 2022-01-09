use super::discrete_image::*;
use super::hashers;
use image::{ImageBuffer, Luma, Rgb};
#[allow(dead_code)]
pub fn rgb8<H: hashers::PixelHasher<[u8; 3], [u8; 3]> + std::marker::Send + std::marker::Sync>(
    img: image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
    precision: [u8; 3],
    hasher: H,
    max_splits: usize,
) -> DiscreteImage<[u8; 3]> {
    let max = img.height().max(img.width());
    let raw_pixels: Vec<[u8; 3]> = img.pixels().map(|f| f.0).collect();
    DiscreteImage::new(
        raw_pixels,
        hasher,
        img.width(),
        precision,
        ((max / 2) as f64).log2() as usize + 1,
        max_splits,
    )
}
#[allow(dead_code)]
pub fn rgb16<
    H: hashers::PixelHasher<[u16; 3], [u16; 3]> + std::marker::Send + std::marker::Sync,
>(
    img: ImageBuffer<Rgb<u16>, Vec<u16>>,
    precision: [u16; 3],
    hasher: H,
    max_splits: usize,
) -> DiscreteImage<[u16; 3]> {
    let max = img.height().max(img.width());
    let raw_pixels: Vec<[u16; 3]> = img.pixels().map(|f| f.0).collect();
    DiscreteImage::new(
        raw_pixels,
        hasher,
        img.width(),
        precision,
        ((max / 2) as f64).log2() as usize + 1,
        max_splits,
    )
}
#[allow(dead_code)]
pub fn luma8<H: hashers::PixelHasher<u8, u8> + std::marker::Send + std::marker::Sync>(
    img: ImageBuffer<Luma<u8>, Vec<u8>>,
    precision: u8,
    hasher: H,
    max_splits: usize,
) -> DiscreteImage<u8> {
    let max = img.height().max(img.width());
    let raw_pixels: Vec<u8> = img.pixels().map(|f| f.0[0]).collect();
    DiscreteImage::new(
        raw_pixels,
        hasher,
        img.width(),
        precision,
        ((max / 2) as f64).log2() as usize + 1,
        max_splits,
    )
}
#[allow(dead_code)]
pub fn luma16<H: hashers::PixelHasher<u16, u16> + std::marker::Send + std::marker::Sync>(
    img: ImageBuffer<Luma<u16>, Vec<u16>>,
    precision: u16,
    hasher: H,
    max_splits: usize,
) -> DiscreteImage<u16> {
    let max = img.height().max(img.width());
    let raw_pixels: Vec<u16> = img.pixels().map(|f| f.0[0]).collect();
    DiscreteImage::new(
        raw_pixels,
        hasher,
        img.width(),
        precision,
        ((max / 2) as f64).log2() as usize + 1,
        max_splits,
    )
}
