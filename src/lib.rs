use image::{self, Primitive};
use num::Integer;
use std::cmp;
trait AsHashedPixel<T: Clone> {
    fn hash(&self) -> T;
}

trait PixelHasher<V, T> {
    fn hash(&self, data: &[Vec<V>], position: (u32, u32), size: (u32, u32)) -> T;
}

impl<T: Clone + Primitive> AsHashedPixel<T> for image::Luma<T> {
    fn hash(&self) -> T {
        self[0]
    }
}

impl AsHashedPixel<[u8; 3]> for image::Rgb<u8> {
    fn hash(&self) -> [u8; 3] {
        [self[0], self[1], self[2]]
    }
}

enum Splitted {
    Horizontal,
    Vertical,
}

enum PixelGroup<T> {
    Leaf(T),
    Node(Box<DiscreteImage<T>>, Box<DiscreteImage<T>>, Splitted),
}

struct DiscreteImage<T> {
    pixels: PixelGroup<T>,
    position: (u32, u32),
    width: u32,
    height: u32,
}

impl<T: std::ops::Sub<Output = T> + std::cmp::PartialOrd + Copy> DiscreteImage<T> {
    pub fn new<V: Clone>(
        raw_data: Vec<impl AsHashedPixel<V>>,
        hasher: impl PixelHasher<V, T>,
        position: (u32, u32),
        size: (u32, u32),
        precision: T,
    ) -> DiscreteImage<T> {
        let array = DiscreteImage::<T>::pixels_to_array(&raw_data, size.0);
        DiscreteImage::create(&array, &hasher, position, size, precision)
    }

    fn pixels_to_array<V: Clone>(pixels: &[impl AsHashedPixel<V>], width: u32) -> Vec<Vec<V>> {
        pixels
            .iter()
            .map(|f| f.hash())
            .collect::<Vec<V>>()
            .chunks_exact(width as usize)
            .map(|row| row.to_vec())
            .collect()
    }

    fn positive_sub(a: T, b: T) -> T {
        match a > b {
            true => a - b,
            false => b - a,
        }
    }

    pub fn group_count(&self) -> usize {
        match &self.pixels {
            PixelGroup::Leaf(_) => 1,
            PixelGroup::Node(f, s, _) => f.group_count() + s.group_count(),
        }
    }

    fn concat(first: &mut Vec<Vec<T>>, mut second: Vec<Vec<T>>, dir: Splitted) {
        match dir {
            Splitted::Horizontal => {
                first.append(&mut second);
            }
            Splitted::Vertical => {
                first
                    .iter_mut()
                    .zip(second)
                    .for_each(|(f, mut s)| f.append(&mut s));
            }
        }
    }

    fn array_from_el(elem: T, width: u32, height: u32) -> Vec<Vec<T>> {
        vec![vec![elem; width as usize]; height as usize]
    }

    pub fn pixels(&self) -> Vec<DiscretePixel<T>> {
        match &self.pixels {
            PixelGroup::Leaf(v) => vec![DiscretePixel::new(
                v.clone(),
                self.position.0,
                self.position.1,
            )],
            PixelGroup::Node(f, s, _) => {
                let mut fp = f.pixels();
                fp.append(&mut s.pixels());
                fp
            }
        }
    }

    pub fn collect(self) -> Vec<T> {
        let (width, height) = (self.width, self.height);
        let array = self.collect_array();
        let mut res = Vec::with_capacity(width as usize * height as usize);
        array.into_iter().for_each(|mut v| res.append(&mut v));
        res
    }

    fn collect_array(self) -> Vec<Vec<T>> {
        match self.pixels {
            PixelGroup::Leaf(el) => DiscreteImage::array_from_el(el, self.width, self.height),
            PixelGroup::Node(f, s, d) => {
                let mut res = f.collect_array();
                DiscreteImage::concat(&mut res, s.collect_array(), d);
                res
            }
        }
    }

    fn create<V: Clone>(
        array: &[Vec<V>],
        hasher: &impl PixelHasher<V, T>,
        position: (u32, u32),
        size: (u32, u32),
        precision: T,
    ) -> DiscreteImage<T> {
        let (width, height) = size;
        let (x, y) = position;
        let pixels: PixelGroup<T> = match size {
            (1, 1) => PixelGroup::Leaf(hasher.hash(array, position, size)),
            _ => match width > height {
                true => {
                    let middle = width / 2;
                    let (left, right) = array.split_at(middle as usize);
                    let (left_hash, right_hash) = (
                        hasher.hash(left, (x, y), (middle, height)),
                        hasher.hash(right, (middle, y), (width - middle, height)),
                    );
                    let eq = DiscreteImage::positive_sub(left_hash, right_hash) > precision;
                    match eq {
                        true => PixelGroup::Leaf(left_hash),
                        false => PixelGroup::Node(
                            Box::new(DiscreteImage::create(
                                left,
                                hasher,
                                (x, y),
                                (middle, height),
                                precision,
                            )),
                            Box::new(DiscreteImage::create(
                                right,
                                hasher,
                                (middle, y),
                                (width - middle, height),
                                precision,
                            )),
                            Splitted::Vertical,
                        ),
                    }
                }
                false => {
                    let middle = height / 2;
                    let (top, bottom) = array.split_at(middle as usize);
                    let (top_hash, bottom_hash) = (
                        hasher.hash(top, (x, y), (width, middle)),
                        hasher.hash(bottom, (x, middle), (width, height - middle)),
                    );
                    let eq = DiscreteImage::positive_sub(top_hash, bottom_hash) > precision;
                    match eq {
                        true => PixelGroup::Leaf(top_hash.into()),
                        false => PixelGroup::Node(
                            Box::new(DiscreteImage::create(
                                top,
                                hasher,
                                (x, y),
                                (width, middle),
                                precision,
                            )),
                            Box::new(DiscreteImage::create(
                                bottom,
                                hasher,
                                (x, middle),
                                (width, height - middle),
                                precision,
                            )),
                            Splitted::Horizontal,
                        ),
                    }
                }
            },
        };
        DiscreteImage {
            pixels,
            position,
            width,
            height,
        }
    }
}

struct DiscretePixel<T> {
    value: T,
    x: u32,
    y: u32,
}

impl<T> DiscretePixel<T> {
    pub fn new(value: T, x: u32, y: u32) -> DiscretePixel<T> {
        DiscretePixel { value, x, y }
    }
}
