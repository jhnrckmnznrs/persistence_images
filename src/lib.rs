pub mod dataset;
pub mod diagram;
pub mod image;
pub mod io;
pub mod optimized;
pub mod interp_fft;

pub use dataset::{SpacingDiagram,load_dataset,to_scaled_diagrams};
pub use diagram::{Diagram,read_diagram};
pub use image::{Bounds,compute_bounds,persistence_image};
pub use optimized::{Engine,SharedGaussianBasis,adaptive_image,batch_images,separable_image,sparse_image};

pub use interp_fft::{InterpFft, StencilCertificate, select_stencil, interp_fft_image};
pub mod stable;
pub use stable::{stable_persistence_image,stable_batch_images};

pub mod stable_basis;
pub use stable_basis::{StableCdfBasis,stable_batch_images_with_engine,stable_batch_images_with_error};

pub mod stable_fft;
pub use stable_fft::{StableFftCertificate,StableIntegratedFft,select_stable_fft_stencil,stable_interp_fft_image,stable_interp_fft_image_with_certificate};
