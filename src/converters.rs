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
