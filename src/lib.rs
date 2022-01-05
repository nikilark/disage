#![feature(test)]
use std::fmt::Debug;
extern crate test;

#[derive(Debug, Clone, Copy, std::cmp::PartialEq, Eq, PartialOrd, Ord)]
pub struct Dimensions {
    pub width: u32,
    pub height: u32,
}

impl Dimensions {
    pub fn new(height: u32, width: u32) -> Dimensions {
        Dimensions { width, height }
    }

    pub fn tuplehw(&self) -> (u32, u32) {
        (self.height, self.width)
    }

    pub fn from_tuplehw((height, width): (u32, u32)) -> Dimensions {
        Dimensions { width, height }
    }
}

#[derive(Debug, Clone, Copy, std::cmp::PartialEq, Eq, PartialOrd, Ord)]
pub struct Position {
    pub x: u32,
    pub y: u32,
}

impl Position {
    pub fn new(x: u32, y: u32) -> Position {
        Position { x, y }
    }

    pub fn tuplexy(&self) -> (u32, u32) {
        (self.x, self.y)
    }

    pub fn from_tuplexy((x, y): (u32, u32)) -> Position {
        Position { x, y }
    }
}

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

    impl AsHashedPixel<[u8; 3]> for image::Rgb<u8> {
        fn hash(&self) -> [u8; 3] {
            [self[0], self[1], self[2]]
        }
    }

    impl AsHashedPixel<[u16; 3]> for image::Rgb<u16> {
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

    impl AsHashedPixel<u8> for image::Luma<u8> {
        fn hash(&self) -> u8 {
            self.clone()[0]
        }
    }

    impl AsHashedPixel<u16> for image::Luma<u16> {
        fn hash(&self) -> u16 {
            self.clone()[0]
        }
    }
}

pub mod hashers {
    use crate::{Dimensions, Position};

    pub trait PixelHasher<V, T> {
        fn hash(&self, data: &[Vec<V>], position: Position, size: Dimensions) -> T;
    }

    pub struct MeanBrightnessHasher {}

    impl PixelHasher<u8, u8> for MeanBrightnessHasher {
        fn hash(&self, data: &[Vec<u8>], position: Position, size: Dimensions) -> u8 {
            let (x, y) = position.tuplexy();
            let (h, w) = size.tuplehw();
            let mut res: u32 = 0;
            for i in y..y + h {
                for j in x..x + w {
                    res += data[i as usize][j as usize].clone() as u32;
                }
            }
            (res / (w * h) as u32) as u8
        }
    }

    impl PixelHasher<u16, u16> for MeanBrightnessHasher {
        fn hash(&self, data: &[Vec<u16>], position: Position, size: Dimensions) -> u16 {
            let (x, y) = position.tuplexy();
            let (h, w) = size.tuplehw();
            let mut res: u64 = 0;
            for i in y..y + h {
                for j in x..x + w {
                    res += data[i as usize][j as usize].clone() as u64;
                }
            }
            (res / (w * h) as u64) as u16
        }
    }

    impl PixelHasher<[u8; 3], [u8; 3]> for MeanBrightnessHasher {
        fn hash(&self, data: &[Vec<[u8; 3]>], position: Position, size: Dimensions) -> [u8; 3] {
            let (x, y) = position.tuplexy();
            let (h, w) = size.tuplehw();
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

    impl PixelHasher<[u16; 3], [u16; 3]> for MeanBrightnessHasher {
        fn hash(&self, data: &[Vec<[u16; 3]>], position: Position, size: Dimensions) -> [u16; 3] {
            let (x, y) = position.tuplexy();
            let (h, w) = size.tuplehw();
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

    pub struct MedianBrightnessHasher {}

    impl PixelHasher<u8, u8> for MedianBrightnessHasher {
        fn hash(&self, data: &[Vec<u8>], position: Position, size: Dimensions) -> u8 {
            let (x, y) = position.tuplexy();
            let (h, w) = size.tuplehw();
            let mut res: Vec<u8> = Vec::with_capacity((h * w) as usize);
            for i in y..y + h {
                for j in x..x + w {
                    res.push(data[i as usize][j as usize].clone());
                }
            }
            res[res.len()/2]
        }
    }

    impl PixelHasher<u16, u16> for MedianBrightnessHasher {
        fn hash(&self, data: &[Vec<u16>], position: Position, size: Dimensions) -> u16 {
            let (x, y) = position.tuplexy();
            let (h, w) = size.tuplehw();
            let mut res: Vec<u16> = Vec::with_capacity((h * w) as usize);
            for i in y..y + h {
                for j in x..x + w {
                    res.push(data[i as usize][j as usize].clone());
                }
            }
            res[res.len()/2]
        }
    }

    impl PixelHasher<[u8; 3], [u8; 3]> for MedianBrightnessHasher {
        fn hash(&self, data: &[Vec<[u8; 3]>], position: Position, size: Dimensions) -> [u8; 3] {
            let (x, y) = position.tuplexy();
            let (h, w) = size.tuplehw();
            let mut res: Vec<[u8; 3]> = Vec::with_capacity((h * w) as usize);
            for i in y..y + h {
                for j in x..x + w {
                    res.push(data[i as usize][j as usize]);
                }
            }
            res.sort_by(|a, b| {
                if a == b {
                    return std::cmp::Ordering::Equal;
                }
                if a[0] < b[0] && a[1] < b[1] && a[2] < b[2] {
                    return std::cmp::Ordering::Less;
                }
                return std::cmp::Ordering::Greater;
            });
            res[res.len()/2]
        }
    }

    impl PixelHasher<[u16; 3], [u16; 3]> for MedianBrightnessHasher {
        fn hash(&self, data: &[Vec<[u16; 3]>], position: Position, size: Dimensions) -> [u16; 3] {
            let (x, y) = position.tuplexy();
            let (h, w) = size.tuplehw();
            let mut res: Vec<[u16; 3]> = Vec::with_capacity((h * w) as usize);
            for i in y..y + h {
                for j in x..x + w {
                    res.push(data[i as usize][j as usize]);
                }
            }
            res.sort_by(|a, b| {
                if a == b {
                    return std::cmp::Ordering::Equal;
                }
                if a[0] < b[0] && a[1] < b[1] && a[2] < b[2] {
                    return std::cmp::Ordering::Less;
                }
                return std::cmp::Ordering::Greater;
            });
            res[res.len()/2]
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

    pub fn to_luma8_from32(array: &[Vec<u32>]) -> image::GrayImage {
        let (h, w) = (array.len(), array[0].len());
        let mut img = image::ImageBuffer::new(w as u32, h as u32);
        img.enumerate_pixels_mut()
            .for_each(|(x, y, pix): (u32, u32, &mut image::Luma<u8>)| {
                pix.0 = [((array[y as usize][x as usize] >> 24) as u8)]
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

pub mod open {
    use super::*;
    use image::{ImageBuffer, Luma, Rgb};
    #[allow(dead_code)]
    pub fn rgb8<
        H: hashers::PixelHasher<[u8; 3], [u8; 3]> + std::marker::Send + std::marker::Sync,
    >(
        img: image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
        precision: [u8; 3],
        hasher: H,
        max_splits : usize
    ) -> DiscreteImage<[u8; 3]> {
        let max = img.height().max(img.width());
        let raw_pixels: Vec<[u8; 3]> = img.pixels().map(|f| f.0).collect();
        DiscreteImage::new(
            raw_pixels,
            hasher,
            img.width(),
            precision,
            ((max/2) as f64).log2() as usize + 1,
            max_splits
        )
    }
    #[allow(dead_code)]
    pub fn rgb16<
        H: hashers::PixelHasher<[u16; 3], [u16; 3]> + std::marker::Send + std::marker::Sync,
    >(
        img: ImageBuffer<Rgb<u16>, Vec<u16>>,
        precision: [u16; 3],
        hasher: H,
        max_splits : usize
    ) -> DiscreteImage<[u16; 3]> {
        let max = img.height().max(img.width());
        let raw_pixels: Vec<[u16; 3]> = img.pixels().map(|f| f.0).collect();
        DiscreteImage::new(
            raw_pixels,
            hasher,
            img.width(),
            precision,
            ((max/2) as f64).log2() as usize + 1,
            max_splits
        )
    }
    #[allow(dead_code)]
    pub fn luma8<H: hashers::PixelHasher<u8, u8> + std::marker::Send + std::marker::Sync>(
        img: ImageBuffer<Luma<u8>, Vec<u8>>,
        precision: u8,
        hasher: H,
        max_splits : usize
    ) -> DiscreteImage<u8> {
        let max = img.height().max(img.width());
        let raw_pixels: Vec<u8> = img.pixels().map(|f| f.0[0]).collect();
        DiscreteImage::new(
            raw_pixels,
            hasher,
            img.width(),
            precision,
            ((max/2) as f64).log2() as usize + 1,
            max_splits
        )
    }
    #[allow(dead_code)]
    pub fn luma16<H: hashers::PixelHasher<u16, u16> + std::marker::Send + std::marker::Sync>(
        img: ImageBuffer<Luma<u16>, Vec<u16>>,
        precision: u16,
        hasher: H,
        max_splits : usize
    ) -> DiscreteImage<u16> {
        let max = img.height().max(img.width());
        let raw_pixels: Vec<u16> = img.pixels().map(|f| f.0[0]).collect();
        DiscreteImage::new(
            raw_pixels,
            hasher,
            img.width(),
            precision,
            ((max/2) as f64).log2() as usize + 1,
            max_splits
        )
    }
}
#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiscretePixel<T> {
    pub value: T,
    pub position: Position,
    pub size: Dimensions,
}

impl<T> DiscretePixel<T> {
    pub fn new(value: T, position: Position, size: Dimensions) -> DiscretePixel<T> {
        DiscretePixel {
            value,
            position,
            size,
        }
    }
}
#[derive(Debug, PartialEq, Clone)]
enum Splitted {
    Horizontal,
    Vertical,
}
#[derive(Debug, PartialEq, Clone)]
enum PixelGroup<T> {
    Leaf(T),
    Node(Box<DiscreteImage<T>>, Box<DiscreteImage<T>>, Splitted),
}
#[derive(Debug, PartialEq, Clone)]
pub struct DiscreteImage<T> {
    pixels: PixelGroup<T>,
    position: Position,
    pub size: Dimensions,
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
        min_splits: usize,
        max_splits : usize
    ) -> DiscreteImage<T> {
        let array = DiscreteImage::<T>::pixels_to_array(&raw_data, width);
        let height = array.len();
        DiscreteImage::create(
            &array,
            &hasher,
            Position::from_tuplexy((0, 0)),
            Dimensions::from_tuplehw((height as u32, width)),
            precision,
            1,
            min_splits, 
            max_splits
        )
    }

    pub fn group_count(&self) -> usize {
        match &self.pixels {
            PixelGroup::Leaf(_) => 1,
            PixelGroup::Node(f, s, _) => f.group_count() + s.group_count(),
        }
    }

    pub fn pixels(&self) -> Vec<DiscretePixel<T>> {
        match &self.pixels {
            PixelGroup::Leaf(v) => vec![DiscretePixel::new(v.clone(), self.position, self.size)],
            PixelGroup::Node(f, s, _) => {
                let mut fp = f.pixels();
                fp.append(&mut s.pixels());
                fp
            }
        }
    }

    pub fn compression(&self) -> i8 {
        (100f64
            - ((self.group_count() as f64 / (self.size.width * self.size.height) as f64)
                * (100f64))) as i8
    }

    pub fn collect(self, borders: Option<T>) -> Vec<Vec<T>> {
        match self.pixels {
            PixelGroup::Leaf(el) => match borders {
                None => DiscreteImage::array_from_el(el, self.size),
                Some(def) => DiscreteImage::array_with_borders(el, self.size, def),
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
        position: Position,
        size: Dimensions,
        precision: T,
        step : usize,
        min_splits : usize,
        max_splits : usize
    ) -> DiscreteImage<T> {
        let (height, width) = size.tuplehw();
        let (x, y) = position.tuplexy();
        let pixels: PixelGroup<T> = match size.tuplehw() {
            (1, 1) => PixelGroup::Leaf(hasher.hash(array, position, size)),
            _ => {
                let ((fpos, fsize), (spos, ssize), split_at) =
                    DiscreteImage::<T>::pre_split(height, width, x, y);
                let f_hash = 
                    hasher.hash(array, fpos, fsize);
                let eq = if step < min_splits {false} else if step > max_splits {true} else  {DiscreteImage::positive_sub(f_hash, hasher.hash(array, spos, ssize)).lt(precision)};
                match eq {
                    true => PixelGroup::Leaf(f_hash),
                    false => {
                        let (f, s) = match height * width > 5000 {
                            true => rayon::join(
                                || {
                                    Box::new(DiscreteImage::create(
                                        array, hasher, fpos, fsize, precision, step + 1, min_splits, max_splits
                                    ))
                                },
                                || {
                                    Box::new(DiscreteImage::create(
                                        array, hasher, spos, ssize, precision, step +1, min_splits, max_splits
                                    ))
                                },
                            ),
                            false => (
                                Box::new(DiscreteImage::create(
                                    array, hasher, fpos, fsize, precision, step + 1, min_splits, max_splits
                                )),
                                Box::new(DiscreteImage::create(
                                    array, hasher, spos, ssize, precision, step+1, min_splits, max_splits
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
            size,
        }
    }

    fn pre_split(
        height: u32,
        width: u32,
        x: u32,
        y: u32,
    ) -> ((Position, Dimensions), (Position, Dimensions), Splitted) {
        match width > height {
            true => {
                let middle = width / 2;
                let f = (
                    Position::from_tuplexy((x, y)),
                    Dimensions::from_tuplehw((height, middle)),
                );
                let s = (
                    Position::from_tuplexy((x + middle, y)),
                    Dimensions::from_tuplehw((height, width - middle)),
                );
                (f, s, Splitted::Vertical)
            }
            false => {
                let middle = height / 2;
                let f = (
                    Position::from_tuplexy((x, y)),
                    Dimensions::from_tuplehw((middle, width)),
                );
                let s = (
                    Position::from_tuplexy((x, y + middle)),
                    Dimensions::from_tuplehw((height - middle, width)),
                );
                (f, s, Splitted::Horizontal)
            }
        }
    }

    fn array_from_el(elem: T, size: Dimensions) -> Vec<Vec<T>> {
        vec![vec![elem; size.width as usize]; size.height as usize]
    }

    fn array_with_borders(elem: T, size: Dimensions, default: T) -> Vec<Vec<T>> {
        let (height, width) = size.tuplehw();
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

    pub fn pixels_to_array<V: Clone>(
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
    use crate::hashers::{MeanBrightnessHasher, PixelHasher};

    use super::*;

    #[test]
    fn array_from_el_test() {
        assert_eq!(
            DiscreteImage::array_from_el(42u8, Dimensions::new(1, 1)),
            vec![vec![42u8]]
        );
        assert_eq!(
            DiscreteImage::array_from_el(42u8, Dimensions::new(2, 2)),
            vec![vec![42u8, 42u8], vec![42u8, 42u8]]
        );
    }

    #[test]
    fn array_with_borders_test() {
        assert_eq!(
            DiscreteImage::array_with_borders(24u8, Dimensions::new(1, 1), 42u8),
            vec![vec![42u8]]
        );
        assert_eq!(
            DiscreteImage::array_with_borders(24u8, Dimensions::new(2, 2), 42u8),
            vec![vec![42u8, 42u8], vec![42u8, 42u8]]
        );
        assert_eq!(
            DiscreteImage::array_with_borders(24u8, Dimensions::new(3, 3), 42u8),
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
        let b = MeanBrightnessHasher {};
        let arr: Vec<Vec<u8>> = vec![
            vec![1, 2, 3],
            vec![4, 5, 6],
            vec![7, 8, 9],
            vec![10, 11, 12],
        ];
        assert_eq!(b.hash(&arr, Position::new(0, 0), Dimensions::new(4, 3)), 6);
        assert_eq!(b.hash(&arr, Position::new(1, 1), Dimensions::new(2, 2)), 7);
        assert_eq!(b.hash(&arr, Position::new(1, 2), Dimensions::new(1, 1)), 8);
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
        let vd = DiscreteImage::create(
            &vimg,
            &MeanBrightnessHasher {},
            Position::new(0, 0),
            Dimensions::new(3, 6),
            2,
            0,
            0,
            1000
        );
        let left = DiscreteImage {
            pixels: PixelGroup::Leaf(1u8),
            position: Position::new(0, 0),
            size: Dimensions::new(3, 3),
        };
        let right = DiscreteImage {
            pixels: PixelGroup::Leaf(22u8),
            position: Position::new(3, 0),
            size: Dimensions::new(3, 3),
        };
        let group = PixelGroup::Node(Box::new(left), Box::new(right), Splitted::Vertical);
        assert_eq!(
            vd,
            DiscreteImage {
                pixels: group,
                position: Position::new(0, 0),
                size: Dimensions::new(3, 6)
            }
        );
        let hd = DiscreteImage::create(
            &himg,
            &MeanBrightnessHasher {},
            Position::new(0, 0),
            Dimensions::new(6, 3),
            2,0,
            0,
            1000
        );
        let top = DiscreteImage {
            pixels: PixelGroup::Leaf(1u8),
            position: Position::new(0, 0),
            size: Dimensions::new(3, 3),
        };
        let bottom = DiscreteImage {
            pixels: PixelGroup::Leaf(22u8),
            position: Position::new(0, 3),
            size: Dimensions::new(3, 3),
        };
        let group = PixelGroup::Node(Box::new(top), Box::new(bottom), Splitted::Horizontal);
        assert_eq!(
            hd,
            DiscreteImage {
                pixels: group,
                position: Position::new(0, 0),
                size: Dimensions::new(6, 3)
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
        let vd = DiscreteImage::create(
            &vimg,
            &MeanBrightnessHasher {},
            Position::new(0, 0),
            Dimensions::new(3, 6),
            2,0,
            0,
            1000
        );
        let left = DiscreteImage {
            pixels: PixelGroup::Leaf(1u8),
            position: Position::new(0, 0),
            size: Dimensions::new(3, 3),
        };
        let right = DiscreteImage {
            pixels: PixelGroup::Leaf(22u8),
            position: Position::new(3, 0),
            size: Dimensions::new(3, 3),
        };
        let group = PixelGroup::Node(Box::new(left), Box::new(right), Splitted::Vertical);
        assert_eq!(
            vd,
            DiscreteImage {
                pixels: group,
                position: Position::new(0, 0),
                size: Dimensions::new(3, 6)
            }
        );
        assert_eq!(vimg, vd.collect(None));
        let hd = DiscreteImage::create(
            &himg,
            &MeanBrightnessHasher {},
            Position::new(0, 0),
            Dimensions::new(6, 3),
            2,0,
            0,
            1000
        );
        let top = DiscreteImage {
            pixels: PixelGroup::Leaf(1u8),
            position: Position::new(0, 0),
            size: Dimensions::new(3, 3),
        };
        let bottom = DiscreteImage {
            pixels: PixelGroup::Leaf(22u8),
            position: Position::new(0, 3),
            size: Dimensions::new(3, 3),
        };
        let group = PixelGroup::Node(Box::new(top), Box::new(bottom), Splitted::Horizontal);
        assert_eq!(
            hd,
            DiscreteImage {
                pixels: group,
                position: Position::new(0, 0),
                size: Dimensions::new(6, 3)
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
            DiscreteImage::new(
                v.clone(),
                hashers::MeanBrightnessHasher {},
                w,
                [200u16, 200, 200],
                10,100000
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
            DiscreteImage::new(v.clone(), hashers::MeanBrightnessHasher {}, w, 100u16, 10,100000);
        });
    }
}
