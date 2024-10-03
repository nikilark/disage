pub(crate) use image;

pub type Position = (u32, u32);
pub type Size = (u32, u32);
pub type Area = (Position, Size);

pub type BaseImage<P> = image::ImageBuffer<P, Vec<<P as image::Pixel>::Subpixel>>;
pub type DiscreteHasher<P> =
    std::sync::Arc<dyn Fn(&BaseImage<P>, Area) -> DiscretePixel<P> + Send + Sync>;
pub type DiscreteEq<P> = std::sync::Arc<dyn Fn(&BaseImage<P>, Area, Area) -> bool + Send + Sync>;

#[derive(Debug, Clone)]
pub struct DiscretePixel<P: image::Pixel> {
    pub pixel: P,
    pub position: Position,
    pub size: Size,
}
