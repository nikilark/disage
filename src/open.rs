use super::*;

pub fn luma_discrete<
    P: 'static + image::Primitive + std::marker::Send + std::marker::Sync,
    T: pixels::PixelOpps<T> + Copy + std::marker::Send + std::marker::Sync,
    H: hashers::PixelHasher<P, T> + std::marker::Send + std::marker::Sync,
    E: checkers::PixelEqChecker<T> + std::marker::Send + std::marker::Sync,
>(
    image: &image::ImageBuffer<image::Luma<P>, Vec<P>>,
    hasher: H,
    equality_checker: E,
    splits: (usize, usize),
) -> discrete_image::DiscreteImage<T> {
    let raw_data = converters::raw_luma(&image);
    DiscreteImage::new(raw_data, hasher, image.width(), equality_checker, splits.0, splits.1)
}

pub fn rgb_discrete<
    P: 'static + image::Primitive + std::marker::Send + std::marker::Sync,
    T: pixels::PixelOpps<T> + Copy + std::marker::Send + std::marker::Sync,
    H: hashers::PixelHasher<[P;3], T> + std::marker::Send + std::marker::Sync,
    E: checkers::PixelEqChecker<T> + std::marker::Send + std::marker::Sync,
>(
    image: &image::ImageBuffer<image::Rgb<P>, Vec<P>>,
    hasher: H,
    equality_checker: E,
    splits: (usize, usize),
) -> discrete_image::DiscreteImage<T> {
    let raw_data = converters::raw_rgb(&image);
    DiscreteImage::new(raw_data, hasher, image.width(), equality_checker, splits.0, splits.1)
}