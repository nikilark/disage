#![feature(test)]
use std::fmt::Debug;
extern crate test;
use rayon::prelude::*;
pub mod pixels {
    pub trait AsHashedPixel<T: Clone> {
        fn hash(&self) -> T;
    }

    impl AsHashedPixel<[u8; 3]> for [u8; 3] {
        fn hash(&self) -> [u8; 3] {
            [self[0], self[1], self[2]]
        }
    }

    impl AsHashedPixel<[u16; 3]> for [u16; 3] {
        fn hash(&self) -> [u16; 3] {
            [self[0], self[1], self[2]]
        }
    }

    impl AsHashedPixel<u8> for u8 {
        fn hash(&self) -> u8 {
            self.clone()
        }
    }

    impl AsHashedPixel<u16> for u16 {
        fn hash(&self) -> u16 {
            self.clone()
        }
    }
}

pub mod hashers {
    use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

    pub trait PixelHasher<V, T> {
        fn hash(&self, data: &[Vec<V>], position: (u32, u32), size: (u32, u32)) -> T;
    }

    pub struct BrightnessHasher {}

    impl PixelHasher<u8, u8> for BrightnessHasher {
        fn hash(&self, data: &[Vec<u8>], position: (u32, u32), size: (u32, u32)) -> u8 {
            let (x, y) = position;
            let (h, w) = size;
            let mut res: u32 = 0;
            for i in y..y + h {
                for j in x..x + w {
                    res += data[i as usize][j as usize].clone() as u32;
                }
            }
            (res / (w * h) as u32) as u8
        }
    }

    impl PixelHasher<u16, u16> for BrightnessHasher {
        fn hash(&self, data: &[Vec<u16>], position: (u32, u32), size: (u32, u32)) -> u16 {
            let (x, y) = position;
            let (h, w) = size;
            let mut res: u64 = 0;
            for i in y..y + h {
                for j in x..x + w {
                    res += data[i as usize][j as usize].clone() as u64;
                }
            }
            (res / (w * h) as u64) as u16
        }
    }

    impl PixelHasher<[u8; 3], [u8; 3]> for BrightnessHasher {
        fn hash(&self, data: &[Vec<[u8; 3]>], position: (u32, u32), size: (u32, u32)) -> [u8; 3] {
            let (x, y) = position;
            let (h, w) = size;
            let mut res: [u32; 3] = [0, 0, 0];
            for i in y..y + h {
                for j in x..x + w {
                    let t = data[i as usize][j as usize];
                    res[0] += t[0] as u32;
                    res[1] += t[1] as u32;
                    res[2] += t[2] as u32;
                }
            }
            let wh = w * h;
            [
                (res[0] / wh) as u8,
                (res[1] / wh) as u8,
                (res[2] / wh) as u8,
            ]
        }
    }

    impl PixelHasher<[u16; 3], [u16; 3]> for BrightnessHasher {
        fn hash(&self, data: &[Vec<[u16; 3]>], position: (u32, u32), size: (u32, u32)) -> [u16; 3] {
            let (x, y) = position;
            let (h, w) = size;
            let mut res: [u64; 3] = [0, 0, 0];
            for i in y..y + h {
                for j in x..x + w {
                    let t = data[i as usize][j as usize];
                    res[0] += t[0] as u64;
                    res[1] += t[1] as u64;
                    res[2] += t[2] as u64;
                }
            }
            let wh = (w * h) as u64;
            [
                (res[0] / wh) as u16,
                (res[1] / wh) as u16,
                (res[2] / wh) as u16,
            ]
        }
    }
}

pub mod converters {
    pub fn to_luma8(array: &[Vec<u8>]) -> image::GrayImage {
        let (h, w) = (array.len(), array[0].len());
        let mut img = image::ImageBuffer::new(w as u32, h as u32);
        img.enumerate_pixels_mut()
            .for_each(|(x, y, p): (u32, u32, &mut image::Luma<u8>)| {
                p.0 = [array[y as usize][x as usize]]
            });
        img
    }

    pub fn to_rgb8(array: &[Vec<[u8; 3]>]) -> image::RgbImage {
        let (h, w) = (array.len(), array[0].len());
        let mut img = image::ImageBuffer::new(w as u32, h as u32);
        img.enumerate_pixels_mut()
            .for_each(|(x, y, pix): (u32, u32, &mut image::Rgb<u8>)| {
                pix.0 = array[y as usize][x as usize]
            });
        img
    }

    pub fn to_rgb16(array: &[Vec<[u16; 3]>]) -> image::ImageBuffer<image::Rgb<u16>, Vec<u16>> {
        let (h, w) = (array.len(), array[0].len());
        let mut img = image::ImageBuffer::new(w as u32, h as u32);
        img.enumerate_pixels_mut()
            .for_each(|(x, y, pix): (u32, u32, &mut image::Rgb<u16>)| {
                pix.0 = array[y as usize][x as usize]
            });
        img
    }

    pub fn to_rgb8_from16(array: &[Vec<[u16; 3]>]) -> image::RgbImage {
        let (h, w) = (array.len(), array[0].len());
        let mut img = image::ImageBuffer::new(w as u32, h as u32);
        img.enumerate_pixels_mut()
            .for_each(|(x, y, pix): (u32, u32, &mut image::Rgb<u8>)| {
                let p = array[y as usize][x as usize];
                let u8pixel = [(p[0] >> 8) as u8, (p[1] >> 8) as u8, (p[2] >> 8) as u8];
                pix.0 = u8pixel;
            });
        img
    }

    pub fn to_luma8_from16(array: &[Vec<u16>]) -> image::GrayImage {
        let (h, w) = (array.len(), array[0].len());
        let mut img = image::ImageBuffer::new(w as u32, h as u32);
        img.enumerate_pixels_mut()
            .for_each(|(x, y, pix): (u32, u32, &mut image::Luma<u8>)| {
                pix.0 = [((array[y as usize][x as usize] >> 8) as u8)]
            });
        img
    }

    pub fn to_luma16(array: &[Vec<u16>]) -> image::ImageBuffer<image::Luma<u16>, Vec<u16>> {
        let (h, w) = (array.len(), array[0].len());
        let mut img = image::ImageBuffer::new(w as u32, h as u32);
        img.enumerate_pixels_mut()
            .for_each(|(x, y, p): (u32, u32, &mut image::Luma<u16>)| {
                p.0 = [array[y as usize][x as usize]]
            });
        img
    }
}

pub struct DiscretePixel<T> {
    value: T,
    x: u32,
    y: u32,
}

impl<T> DiscretePixel<T> {
    pub fn new(value: T, x: u32, y: u32) -> DiscretePixel<T> {
        DiscretePixel { value, x, y }
    }
}
#[derive(Debug, PartialEq)]
enum Splitted {
    Horizontal,
    Vertical,
}
#[derive(Debug, PartialEq)]
enum PixelGroup<T> {
    Leaf(T),
    Node(Box<DiscreteImage<T>>, Box<DiscreteImage<T>>, Splitted),
}
#[derive(Debug, PartialEq)]
pub struct DiscreteImage<T> {
    pixels: PixelGroup<T>,
    position: (u32, u32),
    height: u32,
    width: u32,
}

pub trait PixelOpps<T> {
    fn substract(self, other: T) -> T;
    fn lt(self, other: T) -> bool;
}

impl PixelOpps<[u8; 3]> for [u8; 3] {
    fn substract(self, other: [u8; 3]) -> [u8; 3] {
        [
            self[0].substract(other[0]),
            self[1].substract(other[1]),
            self[2].substract(other[2]),
        ]
    }

    fn lt(self, other: [u8; 3]) -> bool {
        self[0] < other[0] && self[1] < other[1] && self[2] < other[2]
    }
}

impl PixelOpps<[u16; 3]> for [u16; 3] {
    fn substract(self, other: [u16; 3]) -> [u16; 3] {
        [
            self[0].substract(other[0]),
            self[1].substract(other[1]),
            self[2].substract(other[2]),
        ]
    }

    fn lt(self, other: [u16; 3]) -> bool {
        self[0] < other[0] && self[1] < other[1] && self[2] < other[2]
    }
}

impl PixelOpps<u8> for u8 {
    fn substract(self, other: u8) -> u8 {
        match self > other {
            true => self - other,
            false => other - self,
        }
    }

    fn lt(self, other: u8) -> bool {
        self < other
    }
}

impl PixelOpps<u16> for u16 {
    fn substract(self, other: u16) -> u16 {
        match self > other {
            true => self - other,
            false => other - self,
        }
    }

    fn lt(self, other: u16) -> bool {
        self < other
    }
}

impl PixelOpps<i32> for i32 {
    fn substract(self, other: i32) -> i32 {
        match self > other {
            true => self - other,
            false => other - self,
        }
    }

    fn lt(self, other: i32) -> bool {
        self < other
    }
}

impl<T: PixelOpps<T> + Copy + std::marker::Send + std::marker::Sync> DiscreteImage<T> {
    pub fn new<
        V: Clone + Debug + std::marker::Send + std::marker::Sync,
        H: hashers::PixelHasher<V, T> + std::marker::Send + std::marker::Sync,
    >(
        raw_data: Vec<impl pixels::AsHashedPixel<V>>,
        hasher: H,
        width: u32,
        precision: T,
    ) -> DiscreteImage<T> {
        let array = DiscreteImage::<T>::pixels_to_array(&raw_data, width);
        let height = array.len();
        DiscreteImage::create(&array, &hasher, (0, 0), (height as u32, width), precision)
    }

    pub fn group_count(&self) -> usize {
        match &self.pixels {
            PixelGroup::Leaf(_) => 1,
            PixelGroup::Node(f, s, _) => f.group_count() + s.group_count(),
        }
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

    pub fn collect(self, borders: Option<T>) -> Vec<Vec<T>> {
        match self.pixels {
            PixelGroup::Leaf(el) => match borders {
                None => DiscreteImage::array_from_el(el, self.width, self.height),
                Some(def) => DiscreteImage::array_with_borders(el, self.width, self.height, def),
            },
            PixelGroup::Node(f, s, d) => {
                let mut res = f.collect(borders);
                DiscreteImage::concat(&mut res, s.collect(borders), d);
                res
            }
        }
    }

    fn create<
        V: Clone + Debug + std::marker::Send + std::marker::Sync,
        H: hashers::PixelHasher<V, T> + std::marker::Send + std::marker::Sync,
    >(
        array: &[Vec<V>],
        hasher: &H,
        position: (u32, u32),
        size: (u32, u32),
        precision: T,
    ) -> DiscreteImage<T> {
        let (height, width) = size;
        let (x, y) = position;
        let pixels: PixelGroup<T> = match size {
            (1, 1) => PixelGroup::Leaf(hasher.hash(array, position, size)),
            _ => {
                let ((fpos, fsize), (spos, ssize), split_at) =
                    DiscreteImage::<T>::pre_split(height, width, x, y);
                let (f_hash, s_hash) = (
                    hasher.hash(array, fpos, fsize),
                    hasher.hash(array, spos, ssize),
                );
                let eq = DiscreteImage::positive_sub(f_hash, s_hash).lt(precision);
                match eq {
                    true => PixelGroup::Leaf(s_hash),
                    false => {
                        let (f, s) = match height * width > 5000 {
                            true => rayon::join(
                                || {
                                    Box::new(DiscreteImage::create(
                                        array, hasher, fpos, fsize, precision,
                                    ))
                                },
                                || {
                                    Box::new(DiscreteImage::create(
                                        array, hasher, spos, ssize, precision,
                                    ))
                                },
                            ),
                            false => (
                                Box::new(DiscreteImage::create(
                                    array, hasher, fpos, fsize, precision,
                                )),
                                Box::new(DiscreteImage::create(
                                    array, hasher, spos, ssize, precision,
                                )),
                            ),
                        };
                        PixelGroup::Node(f, s, split_at)
                    }
                }
            }
        };
        DiscreteImage {
            pixels,
            position,
            width,
            height,
        }
    }

    fn pre_split(
        height: u32,
        width: u32,
        x: u32,
        y: u32,
    ) -> (((u32, u32), (u32, u32)), ((u32, u32), (u32, u32)), Splitted) {
        match width > height {
            true => {
                let middle = width / 2;
                let f = ((x, y), (height, middle));
                let s = ((x + middle, y), (height, width - middle));
                (f, s, Splitted::Vertical)
            }
            false => {
                let middle = height / 2;
                let f = ((x, y), (middle, width));
                let s = ((x, y + middle), (height - middle, width));
                (f, s, Splitted::Horizontal)
            }
        }
    }

    fn array_from_el(elem: T, width: u32, height: u32) -> Vec<Vec<T>> {
        vec![vec![elem; width as usize]; height as usize]
    }

    fn array_with_borders(elem: T, width: u32, height: u32, default: T) -> Vec<Vec<T>> {
        let mut res = Vec::with_capacity(height as usize);
        res.push(vec![default; width as usize]);
        for _ in 1..height - 1 {
            let mut row = Vec::with_capacity(width as usize);
            row.push(default);
            if width > 2 {
                row.append(&mut vec![elem; (width - 2) as usize]);
            }
            row.push(default);
            res.push(row);
        }
        if height > 1 {
            res.push(vec![default; width as usize]);
        }
        res
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

    fn pixels_to_array<V: Clone>(
        pixels: &[impl pixels::AsHashedPixel<V>],
        width: u32,
    ) -> Vec<Vec<V>> {
        pixels
            .iter()
            .map(|f| f.hash())
            .collect::<Vec<V>>()
            .chunks_exact(width as usize)
            .map(|row| row.to_vec())
            .collect()
    }

    fn positive_sub(a: T, b: T) -> T {
        a.substract(b)
    }
}
#[cfg(test)]
mod tests {
    use crate::hashers::{BrightnessHasher, PixelHasher};

    use super::*;

    #[test]
    fn array_from_el_test() {
        assert_eq!(DiscreteImage::array_from_el(42u8, 1, 1), vec![vec![42u8]]);
        assert_eq!(
            DiscreteImage::array_from_el(42u8, 2, 2),
            vec![vec![42u8, 42u8], vec![42u8, 42u8]]
        );
    }

    #[test]
    fn array_with_borders_test() {
        assert_eq!(
            DiscreteImage::array_with_borders(24u8, 1, 1, 42u8),
            vec![vec![42u8]]
        );
        assert_eq!(
            DiscreteImage::array_with_borders(24u8, 2, 2, 42u8),
            vec![vec![42u8, 42u8], vec![42u8, 42u8]]
        );
        assert_eq!(
            DiscreteImage::array_with_borders(24u8, 3, 3, 42u8),
            vec![
                vec![42u8, 42u8, 42u8],
                vec![42u8, 24u8, 42u8],
                vec![42u8, 42u8, 42u8]
            ]
        );
    }

    #[test]
    fn concat_test() {
        let mut first = vec![vec![1, 2, 3], vec![4, 5, 6]];
        let mut f2 = first.clone();
        let second = vec![vec![7, 8, 9], vec![10, 11, 12]];
        let s2 = second.clone();
        let splitted_vertical = vec![vec![1, 2, 3, 7, 8, 9], vec![4, 5, 6, 10, 11, 12]];
        let splitted_horizontal = vec![
            vec![1, 2, 3],
            vec![4, 5, 6],
            vec![7, 8, 9],
            vec![10, 11, 12],
        ];
        DiscreteImage::concat(&mut first, second, Splitted::Vertical);
        assert_eq!(first, splitted_vertical);
        DiscreteImage::concat(&mut f2, s2, Splitted::Horizontal);
        assert_eq!(f2, splitted_horizontal);
        let mut single1 = vec![vec![1]];
        let mut single2 = vec![vec![1]];
        DiscreteImage::concat(&mut single1, vec![vec![1]], Splitted::Vertical);
        assert_eq!(single1, vec![vec![1, 1]]);
        DiscreteImage::concat(&mut single2, vec![vec![1]], Splitted::Horizontal);
        assert_eq!(single2, vec![vec![1], vec![1]]);
    }

    #[test]
    fn pixels_to_array_test() {
        assert_eq!(
            DiscreteImage::<u8>::pixels_to_array(&vec![42u8], 1),
            vec![vec![42u8]]
        );
        assert_eq!(
            DiscreteImage::<u8>::pixels_to_array(&vec![42u8; 9], 3),
            vec![
                vec![42u8, 42u8, 42u8],
                vec![42u8, 42u8, 42u8],
                vec![42u8, 42u8, 42u8]
            ]
        );
        assert_eq!(
            DiscreteImage::<u8>::pixels_to_array(&vec![42u8; 12], 3),
            vec![
                vec![42u8, 42u8, 42u8],
                vec![42u8, 42u8, 42u8],
                vec![42u8, 42u8, 42u8],
                vec![42u8, 42u8, 42u8]
            ]
        );
        assert_eq!(
            DiscreteImage::<u8>::pixels_to_array(&vec![42u8; 12], 4),
            vec![
                vec![42u8, 42u8, 42u8, 42u8],
                vec![42u8, 42u8, 42u8, 42u8],
                vec![42u8, 42u8, 42u8, 42u8]
            ]
        );
    }

    #[test]
    fn positive_sub_test() {
        assert_eq!(
            DiscreteImage::positive_sub(10, 20),
            DiscreteImage::positive_sub(20, 10)
        );
    }

    #[test]
    fn brightness_hasher_test() {
        let b = BrightnessHasher {};
        let arr: Vec<Vec<u8>> = vec![
            vec![1, 2, 3],
            vec![4, 5, 6],
            vec![7, 8, 9],
            vec![10, 11, 12],
        ];
        assert_eq!(b.hash(&arr, (0, 0), (4, 3)), 6);
        assert_eq!(b.hash(&arr, (1, 1), (2, 2)), 7);
        assert_eq!(b.hash(&arr, (1, 2), (1, 1)), 8);
    }

    #[test]
    fn create_test() {
        let mut arr1: Vec<Vec<u8>> = vec![vec![1, 1, 1], vec![1, 1, 1], vec![1, 1, 1]];
        let arr2: Vec<Vec<u8>> = vec![vec![22, 22, 22], vec![22, 22, 22], vec![22, 22, 22]];
        let mut arr1c = arr1.clone();
        DiscreteImage::concat(&mut arr1, arr2.clone(), Splitted::Vertical);
        DiscreteImage::concat(&mut arr1c, arr2, Splitted::Horizontal);
        let vimg = arr1;
        let himg = arr1c;
        let vd = DiscreteImage::create(&vimg, &BrightnessHasher {}, (0, 0), (3, 6), 2);
        let left = DiscreteImage {
            pixels: PixelGroup::Leaf(1u8),
            position: (0, 0),
            width: 3,
            height: 3,
        };
        let right = DiscreteImage {
            pixels: PixelGroup::Leaf(22u8),
            position: (3, 0),
            width: 3,
            height: 3,
        };
        let group = PixelGroup::Node(Box::new(left), Box::new(right), Splitted::Vertical);
        assert_eq!(
            vd,
            DiscreteImage {
                pixels: group,
                position: (0, 0),
                width: 6,
                height: 3
            }
        );
        let hd = DiscreteImage::create(&himg, &BrightnessHasher {}, (0, 0), (6, 3), 2);
        let top = DiscreteImage {
            pixels: PixelGroup::Leaf(1u8),
            position: (0, 0),
            width: 3,
            height: 3,
        };
        let bottom = DiscreteImage {
            pixels: PixelGroup::Leaf(22u8),
            position: (0, 3),
            width: 3,
            height: 3,
        };
        let group = PixelGroup::Node(Box::new(top), Box::new(bottom), Splitted::Horizontal);
        assert_eq!(
            hd,
            DiscreteImage {
                pixels: group,
                position: (0, 0),
                width: 3,
                height: 6
            }
        );
    }

    #[test]
    fn collect_test() {
        let mut arr1: Vec<Vec<u8>> = vec![vec![1, 1, 1], vec![1, 1, 1], vec![1, 1, 1]];
        let arr2: Vec<Vec<u8>> = vec![vec![22, 22, 22], vec![22, 22, 22], vec![22, 22, 22]];
        let mut arr1c = arr1.clone();
        DiscreteImage::concat(&mut arr1, arr2.clone(), Splitted::Vertical);
        DiscreteImage::concat(&mut arr1c, arr2, Splitted::Horizontal);
        let vimg = arr1;
        let himg = arr1c;
        let vd = DiscreteImage::create(&vimg, &BrightnessHasher {}, (0, 0), (3, 6), 2);
        let left = DiscreteImage {
            pixels: PixelGroup::Leaf(1u8),
            position: (0, 0),
            width: 3,
            height: 3,
        };
        let right = DiscreteImage {
            pixels: PixelGroup::Leaf(22u8),
            position: (3, 0),
            width: 3,
            height: 3,
        };
        let group = PixelGroup::Node(Box::new(left), Box::new(right), Splitted::Vertical);
        assert_eq!(
            vd,
            DiscreteImage {
                pixels: group,
                position: (0, 0),
                width: 6,
                height: 3
            }
        );
        assert_eq!(vimg, vd.collect(None));
        let hd = DiscreteImage::create(&himg, &BrightnessHasher {}, (0, 0), (6, 3), 2);
        let top = DiscreteImage {
            pixels: PixelGroup::Leaf(1u8),
            position: (0, 0),
            width: 3,
            height: 3,
        };
        let bottom = DiscreteImage {
            pixels: PixelGroup::Leaf(22u8),
            position: (0, 3),
            width: 3,
            height: 3,
        };
        let group = PixelGroup::Node(Box::new(top), Box::new(bottom), Splitted::Horizontal);
        assert_eq!(
            hd,
            DiscreteImage {
                pixels: group,
                position: (0, 0),
                width: 3,
                height: 6
            }
        );
        assert_eq!(himg, hd.collect(None));
    }

    #[bench]
    fn open_discrete_rgb16(b: &mut test::Bencher) {
        let img = image::io::Reader::open("./test/test.jpg")
            .expect("Failed to open image")
            .decode()
            .expect("Failed to decode image")
            .into_rgb16();
        let w = img.width();
        let v: Vec<[u16; 3]> = img.pixels().map(|f| f.0).collect();
        b.iter(move || {
            let discrete = DiscreteImage::new(
                v.clone(),
                hashers::BrightnessHasher {},
                w,
                [200u16, 200, 200],
            );
        });
    }

    #[bench]
    fn open_discrete_luma16(b: &mut test::Bencher) {
        let img = image::io::Reader::open("./test/test.jpg")
            .expect("Failed to open image")
            .decode()
            .expect("Failed to decode image")
            .into_luma16();
        let w = img.width();
        let v: Vec<u16> = img.pixels().map(|f| f.0[0]).collect();
        b.iter(move || {
            let discrete = DiscreteImage::new(v.clone(), hashers::BrightnessHasher {}, w, 100u16);
        });
    }
}
