use super::{checkers, converters, hashers, pixels};
use std::fmt::Debug;

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
pub enum Splitted {
    Horizontal,
    Vertical,
}
#[derive(Debug, PartialEq, Clone)]
pub enum PixelGroup<T> {
    Leaf(T),
    Node(Box<DiscreteImage<T>>, Box<DiscreteImage<T>>, Splitted),
}
#[derive(Debug, PartialEq, Clone)]
pub struct DiscreteImage<T> {
    pixels: PixelGroup<T>,
    position: Position,
    pub size: Dimensions,
}

impl<T: pixels::PixelOpps<T> + Copy + std::marker::Send + std::marker::Sync + std::fmt::Debug>
    DiscreteImage<T>
{
    pub fn new<
        V: Clone + std::marker::Send + std::marker::Sync,
        H: hashers::PixelHasher<V, T> + std::marker::Send + std::marker::Sync,
        E: checkers::PixelEqChecker<T> + std::marker::Send + std::marker::Sync,
    >(
        raw_data: Vec<V>,
        hasher: H,
        width: u32,
        eq_checher: E,
        min_splits: usize,
        max_splits: usize,
    ) -> DiscreteImage<T> {
        let array = converters::pixels_to_array(&raw_data, width);
        let height = array.len();
        DiscreteImage::create(
            &array,
            &hasher,
            Position::from_tuplexy((0, 0)),
            Dimensions::from_tuplehw((height as u32, width)),
            &eq_checher,
            1,
            min_splits,
            max_splits,
        )
    }

    pub fn contains(&self, pos: Position) -> bool {
        let (x, y) = self.position.tuplexy();
        let (h, w) = self.size.tuplehw();
        let (dx, dy) = pos.tuplexy();
        x <= dx && dx < x + w && y <= dy && dy < y + h
    }

    pub fn pixel_at(&self, pos: Position) -> Option<DiscretePixel<T>> {
        if !self.contains(pos) {
            return None;
        }
        match &self.pixels {
            PixelGroup::Leaf(v) => Some(DiscretePixel {
                value: v.clone(),
                position: self.position,
                size: self.size,
            }),
            PixelGroup::Node(l, r, _) => (if l.contains(pos) { l } else { r }).pixel_at(pos),
        }
    }

    pub fn modify_leaf(&mut self, value: T, pos: Position) {
        if !self.contains(pos) {
            return;
        }
        match &mut self.pixels {
            PixelGroup::Leaf(v) => *v = value,
            PixelGroup::Node(l, r, _) => {
                (if l.contains(pos) { l } else { r }).modify_leaf(value, pos)
            }
        }
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

    pub fn pixels_mut(&mut self) -> Vec<DiscretePixel<&mut T>> {
        match &mut self.pixels {
            PixelGroup::Leaf(v) => vec![DiscretePixel::new(v, self.position, self.size)],
            PixelGroup::Node(f, s, _) => {
                let mut fp = f.pixels_mut();
                fp.append(&mut s.pixels_mut());
                fp
            }
        }
    }

    pub fn compression(&self) -> i8 {
        (100f64
            - ((self.group_count() as f64 / (self.size.width * self.size.height) as f64)
                * (100f64))) as i8
    }

    pub fn collect(self) -> Vec<Vec<T>> {
        match self.pixels {
            PixelGroup::Leaf(el) => DiscreteImage::array_from_el(el, self.size),
            PixelGroup::Node(f, s, d) => {
                let mut res = f.collect();
                DiscreteImage::concat(&mut res, s.collect(), d);
                res
            }
        }
    }

    pub fn collect_with_borders(self, borders: T) -> Vec<Vec<T>> {
        match self.pixels {
            PixelGroup::Leaf(el) => DiscreteImage::array_with_borders(el, self.size, borders),
            PixelGroup::Node(f, s, d) => {
                let mut res = f.collect_with_borders(borders);
                DiscreteImage::concat(&mut res, s.collect_with_borders(borders), d);
                res
            }
        }
    }

    fn create<
        V: Clone + std::marker::Send + std::marker::Sync,
        H: hashers::PixelHasher<V, T> + std::marker::Send + std::marker::Sync,
        E: checkers::PixelEqChecker<T> + std::marker::Send + std::marker::Sync,
    >(
        array: &[Vec<V>],
        hasher: &H,
        position: Position,
        size: Dimensions,
        eq_checher: &E,
        step: usize,
        min_splits: usize,
        max_splits: usize,
    ) -> DiscreteImage<T> {
        let (height, width) = size.tuplehw();
        let (x, y) = position.tuplexy();
        let pixels: PixelGroup<T> = match size.tuplehw() {
            (1, 1) => PixelGroup::Leaf(hasher.hash(array, position, size)),
            (1, 2) => PixelGroup::Leaf(hasher.hash(array, position, size)),
            (2, 1) => PixelGroup::Leaf(hasher.hash(array, position, size)),
            (2, 2) => PixelGroup::Leaf(hasher.hash(array, position, size)),
            _ => {
                let (pref_dir, other_dir) = if width > height {
                    (Splitted::Vertical, Splitted::Horizontal)
                } else {
                    (Splitted::Horizontal, Splitted::Vertical)
                };
                let ((mut fpos, mut fsize), (mut spos, mut ssize), mut split_at) =
                    DiscreteImage::<T>::pre_split(height, width, x, y, pref_dir);
                let mut f_hash = hasher.hash(&array, fpos, fsize);
                let eq = if step > max_splits {
                    true
                } else if step < min_splits {
                    false
                } else {
                    if eq_checher.eq(f_hash, hasher.hash(&array, spos, ssize)) {
                        if height > 1 && width > 1 {
                            let ((fpos2, fsize2), (spos2, ssize2), split_at2) =
                                DiscreteImage::<T>::pre_split(height, width, x, y, other_dir);
                            let f_hash2 = hasher.hash(&array, fpos2, fsize2);
                            if eq_checher.eq(f_hash2, hasher.hash(&array, spos2, ssize2)) {
                                true
                            } else {
                                ((fpos, fsize), (spos, ssize), split_at) =
                                    ((fpos2, fsize2), (spos2, ssize2), split_at2);
                                f_hash = f_hash2;
                                false
                            }
                        } else {
                            true
                        }
                    } else {
                        false
                    }
                };
                match eq {
                    true => PixelGroup::Leaf(f_hash),
                    false => {
                        let (f, s) = match height * width > 5000 {
                            true => rayon::join(
                                || {
                                    Box::new(DiscreteImage::create(
                                        array,
                                        hasher,
                                        fpos,
                                        fsize,
                                        eq_checher,
                                        step + 1,
                                        min_splits,
                                        max_splits,
                                    ))
                                },
                                || {
                                    Box::new(DiscreteImage::create(
                                        array,
                                        hasher,
                                        spos,
                                        ssize,
                                        eq_checher,
                                        step + 1,
                                        min_splits,
                                        max_splits,
                                    ))
                                },
                            ),
                            false => (
                                Box::new(DiscreteImage::create(
                                    array,
                                    hasher,
                                    fpos,
                                    fsize,
                                    eq_checher,
                                    step + 1,
                                    min_splits,
                                    max_splits,
                                )),
                                Box::new(DiscreteImage::create(
                                    array,
                                    hasher,
                                    spos,
                                    ssize,
                                    eq_checher,
                                    step + 1,
                                    min_splits,
                                    max_splits,
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
        dir: Splitted,
    ) -> ((Position, Dimensions), (Position, Dimensions), Splitted) {
        match dir {
            Splitted::Vertical => {
                let middle = width / 2;
                if middle == 0 {
                    println!("Vertical {:?}, {:?}", (height, width), (x, y))
                }
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
            Splitted::Horizontal => {
                let middle = height / 2;
                if middle == 0 {
                    println!("Horizontal {:?}, {:?}", (height, width), (x, y))
                }
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
}

#[cfg(test)]
mod tests {
    use super::hashers::*;
    use super::*;
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
            &checkers::BrightnessChecker { precision: 2 },
            0,
            0,
            1000,
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
            &checkers::BrightnessChecker { precision: 2 },
            0,
            0,
            1000,
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
            &checkers::BrightnessChecker { precision: 2 },
            0,
            0,
            1000,
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
        assert_eq!(vimg, vd.collect());
        let hd = DiscreteImage::create(
            &himg,
            &MeanBrightnessHasher {},
            Position::new(0, 0),
            Dimensions::new(6, 3),
            &checkers::BrightnessChecker { precision: 2 },
            0,
            0,
            1000,
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
        assert_eq!(himg, hd.collect());
    }

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
}
