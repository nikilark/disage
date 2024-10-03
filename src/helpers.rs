pub(crate) use num_traits::cast::{NumCast, ToPrimitive};

use crate::types::*;

pub fn average_hasher<P: image::Pixel>(image: &BaseImage<P>, area: Area) -> DiscretePixel<P> {
    let (position, size) = area;
    let (x, y) = position;
    let (x_end, y_end) = (x + size.0, y + size.1);

    let mut sum: Vec<u128> = vec![0; P::CHANNEL_COUNT as usize];

    for x in x..x_end {
        for y in y..y_end {
            let pixel = image.get_pixel(x, y);
            for i in 0..P::CHANNEL_COUNT as usize {
                sum[i] += pixel.channels()[i].to_u128().unwrap();
            }
        }
    }

    let average_to = size.0 * size.1;
    let mut pixel = image.get_pixel(0, 0).clone();
    for i in 0..P::CHANNEL_COUNT as usize {
        pixel.channels_mut()[i] =
            <P::Subpixel as NumCast>::from(sum[i] / average_to as u128).unwrap();
    }

    DiscretePixel {
        pixel,
        position,
        size,
    }
}

pub fn random_eq<P: image::Pixel>(
    image: &BaseImage<P>,
    first: Area,
    second: Area,
    precision: P::Subpixel,
) -> bool {
    let (position_first, size_first) = first;
    let (position_second, size_second) = second;

    let pixels_to_check = (size_first.0 * size_first.1).ilog2();
    let max_width = size_first.0.min(size_second.0);
    let max_height = size_first.1.min(size_second.1);

    use rand::Rng;
    let mut rng = rand::thread_rng();

    use std::ops::Sub;
    for _ in 0..pixels_to_check {
        let random_x = rng.gen::<u32>() % max_width;
        let random_y = rng.gen::<u32>() % max_height;

        let pixel_first = image.get_pixel(position_first.0 + random_x, position_first.1 + random_y);
        let pixel_second =
            image.get_pixel(position_second.0 + random_x, position_second.1 + random_y);

        for i in 0..P::CHANNEL_COUNT as usize {
            if (pixel_first.channels()[i] > pixel_second.channels()[i])
                && (pixel_first.channels()[i].sub(pixel_second.channels()[i]) > precision)
                || (pixel_second.channels()[i] > pixel_first.channels()[i])
                    && (pixel_second.channels()[i].sub(pixel_first.channels()[i]) > precision)
            {
                return false;
            }
        }
    }
    true
}
