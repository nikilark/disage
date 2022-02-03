use super::pixels;

pub trait PixelEqChecker<T> {
    fn eq(&self, left : T, right : T) -> bool;
}

pub struct BrightnessChecker<P : pixels::PixelOpps<P>>
{
    pub precision : P
}

impl<P:pixels::PixelOpps<P> + Clone> PixelEqChecker<P> for BrightnessChecker<P>
{
    fn eq(&self, left : P, right : P) -> bool {
        return left.substract(right).lt(self.precision.clone());
    }
}