use crate::helpers::*;
use crate::types::*;

pub struct DiscreteImage<'a, P: image::Pixel> {
    pub base: &'a BaseImage<P>,
    pub pixels: Vec<DiscretePixel<P>>,
    hash: DiscreteHasher<P>,
    eq: DiscreteEq<P>,
    min_iterations: u32,
}

impl<'a, P> DiscreteImage<'a, P>
where
    P: image::Pixel + 'static + Send + Sync,
    P::Subpixel: Sync,
{
    pub fn new(base: &'a BaseImage<P>) -> DiscreteImage<'a, P> {
        use image::Primitive;

        DiscreteImage::new_custom(
            base,
            std::sync::Arc::new(average_hasher),
            std::sync::Arc::new(
                |image: &image::ImageBuffer<P, Vec<P::Subpixel>>, first, second| {
                    random_eq(
                        image,
                        first,
                        second,
                        <P::Subpixel as NumCast>::from(
                            P::Subpixel::DEFAULT_MAX_VALUE.to_f64().unwrap() * 0.05 as f64,
                        )
                        .unwrap(),
                    )
                },
            ),
        )
    }

    pub fn new_custom(
        base: &'a BaseImage<P>,
        hasher: DiscreteHasher<P>,
        comparator: DiscreteEq<P>,
    ) -> DiscreteImage<'a, P> {
        let min_iterations = base.dimensions().0.min(base.dimensions().1).ilog2() as u32;

        let mut this = DiscreteImage {
            base,
            pixels: Vec::new(),
            hash: hasher,
            eq: comparator,
            min_iterations,
        };

        this.pixels = this.disage(((0, 0), base.dimensions()), 0);
        this
    }

    pub fn collect(&self) -> BaseImage<P> {
        self.collect_with_borders(None)
    }

    pub fn collect_with_borders(&self, border: Option<P>) -> BaseImage<P> {
        let mut img = self.base.clone();
        for pixel in self.pixels.iter() {
            let (x_start, y_start) = pixel.position;
            let (x_end, y_end) = (x_start + pixel.size.0, y_start + pixel.size.1);

            for x in x_start..x_end {
                for y in y_start..y_end {
                    if (x == x_start || x == x_end - 1 || y == y_start || y == y_end - 1)
                        && border.is_some()
                    {
                        img.put_pixel(x, y, border.unwrap());
                    } else {
                        img.put_pixel(x, y, pixel.pixel);
                    }
                }
            }
        }
        img
    }

    fn disage(&self, area: Area, iteration: u32) -> Vec<DiscretePixel<P>> {
        let (position, size) = area;
        let (width, height) = size;

        if width == 1 || height == 1 {
            return vec![(self.hash)(self.base, (position, size))];
        }

        let horizontal_priority = width < height;
        let branch = self.branch(area, horizontal_priority, iteration);
        if let Some((first, second)) = branch {
            [first, second].concat()
        } else {
            vec![(self.hash)(self.base, (position, size))]
        }
    }

    fn branch(
        &self,
        area: Area,
        horizontal_priority: bool,
        iteration: u32,
    ) -> Option<(Vec<DiscretePixel<P>>, Vec<DiscretePixel<P>>)> {
        let prioritized_split = self.split(area, horizontal_priority);
        if !(self.eq)(self.base, prioritized_split.0, prioritized_split.1) {
            return Some(rayon::join(
                || self.disage(prioritized_split.0, iteration + 1),
                || self.disage(prioritized_split.1, iteration + 1),
            ));
        }

        let non_prioritized_split = self.split(area, !horizontal_priority);
        if !(self.eq)(self.base, non_prioritized_split.0, non_prioritized_split.1) {
            return Some(rayon::join(
                || self.disage(non_prioritized_split.0, iteration + 1),
                || self.disage(non_prioritized_split.1, iteration + 1),
            ));
        }

        if iteration < self.min_iterations {
            return Some(rayon::join(
                || self.disage(prioritized_split.0, iteration + 1),
                || self.disage(prioritized_split.1, iteration + 1),
            ));
        }
        None
    }

    fn split(&self, area: Area, horizontal: bool) -> (Area, Area) {
        let (position, size) = area;
        let (width, height) = size;

        let size_first = if horizontal {
            (width, height / 2)
        } else {
            (width / 2, height)
        };
        let size_second = if horizontal {
            (width, height - size_first.1)
        } else {
            (width - size_first.0, height)
        };

        let position_first = position;
        let position_second = if horizontal {
            (position.0, position.1 + size_first.1)
        } else {
            (position.0 + size_first.0, position.1)
        };

        ((position_first, size_first), (position_second, size_second))
    }
}
