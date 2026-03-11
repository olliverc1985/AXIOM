//! Text input pipeline — tokeniser and encoder for converting raw text to tensors.

pub mod encoder;
pub mod tokeniser;

pub use encoder::Encoder;
pub use tokeniser::Tokeniser;
