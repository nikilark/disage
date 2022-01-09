use super::discrete_image::{Dimensions, Position};

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
        res[res.len() / 2]
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
        res[res.len() / 2]
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
        res[res.len() / 2]
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
        res[res.len() / 2]
    }
}
