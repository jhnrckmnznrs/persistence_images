//! Error-certified one-FFT approximation for weighted, pixel-INTEGRATED persistence images.
//!
//! The certificate controls interpolation truncation in exact arithmetic, not
//! floating-point deposition, cancellation, or FFT roundoff. The lattice is
//! extended with ghost nodes so input points on the domain boundary are valid.
use crate::diagram::Diagram;
use crate::image::Bounds;
use crate::stable::cdf_interval;
use std::f64::consts::PI;
use rustfft::{Fft, FftPlanner, num_complex::Complex};
use std::sync::Arc;

const MAX_STENCIL_WIDTH: usize = 18;
const MAX_FFT_CELLS: usize = 4_000_000;

#[derive(Debug, Clone, Copy)]
pub struct StableFftCertificate {
    pub width_x: usize,
    pub width_y: usize,
    /// Uniform per-pixel interpolation error in exact arithmetic.
    pub pixel_error: f64,
    /// Uniform whole-image L1 interpolation bound, independent of pixel count.
    pub vector_l1_error: f64,
}

fn log_factorial(n: usize) -> f64 {
    (2..=n).map(|i| (i as f64).ln()).sum()
}

/// One-dimensional pixel-mass and whole-axis L1 Lagrange error bounds.
/// Derived from Gaussian derivatives, Hermite polynomials and the exact
/// maximal central-node distance product. No FFT roundoff is included.
fn axis_errors(r:usize,h:f64,sigma:f64)->(f64,f64) {
    let t=h/sigma;
    if !t.is_finite()||t<=0.0{return(f64::INFINITY,f64::INFINITY);}
    let rf=r as f64;
    let common=2.0*log_factorial(2*r)-3.0*log_factorial(r)
        -5.0*rf*std::f64::consts::LN_2+2.0*rf*t.ln();
    let pixel=common+(t/(2.0*PI).sqrt()).ln();
    let vector=1.5*log_factorial(2*r)-2.0*log_factorial(r)
        -4.0*rf*std::f64::consts::LN_2+2.0*rf*t.ln();
    (if pixel>700.0{f64::INFINITY}else{pixel.exp()},
     if vector>700.0{f64::INFINITY}else{vector.exp()})
}

fn fast_len(n: usize) -> Option<usize> {
    // rustfft supports mixed-radix FFTs. Choose 2/3/5-smooth sizes to
    // avoid excessive power-of-two padding for typical image dimensions.
    let mut k = n;
    loop {
        let mut v = k;
        for factor in [2,3,5] {
            while v > 1 && v % factor == 0 { v /= factor; }
        }
        if v == 1 { return Some(k); }
        k = k.checked_add(1)?;
    }
}

/// Returns FFT dimensions for a source lattice with ghost-node padding.
pub fn fft_shape(nx:usize, ny:usize, wx:usize, wy:usize)->Option<(usize,usize)> {
    let px = wx.checked_add(2)?.max(3);
    let py = wy.checked_add(2)?.max(3);
    let mx = nx.checked_add(px.checked_mul(2)?)?;
    let my = ny.checked_add(py.checked_mul(2)?)?;
    let lx = fast_len(mx.checked_add(nx)?.checked_sub(1)?)?;
    let ly = fast_len(my.checked_add(ny)?.checked_sub(1)?)?;
    if lx.checked_mul(ly)? > MAX_FFT_CELLS { None } else { Some((lx,ly)) }
}

/// Choose an even stencil meeting the uniform pixel-error budget.
/// weight_sum is the sum of nonnegative ramp weights (NOT the point count).
/// No width within the supported range => None, not an uncertified answer.
pub fn select_stable_fft_stencil(weight_sum:f64,bounds:Bounds,nx:usize,ny:usize,
        sx:f64,sy:f64,epsilon:f64)->Result<Option<StableFftCertificate>,String>{
    if nx==0||ny==0||nx.checked_mul(ny).is_none(){
        return Err("Grid dimensions must be positive and representable".into());
    }
    if !epsilon.is_finite()||epsilon<=0.0
        ||!weight_sum.is_finite()||weight_sum<0.0 {
        return Err("Error budget and total weight must be finite and valid".into());
    }
    if !(sx.is_finite()&&sx>0.0&&sy.is_finite()&&sy>0.0
        &&bounds.x_min.is_finite()&&bounds.x_max.is_finite()
        &&bounds.y_min.is_finite()&&bounds.y_max.is_finite()
        &&bounds.x_min<bounds.x_max&&bounds.y_min<bounds.y_max) {
        return Err("Invalid bounds or Gaussian bandwidth".into());
    }
    let hx=(bounds.x_max-bounds.x_min)/nx as f64;
    let hy=(bounds.y_max-bounds.y_min)/ny as f64;
    if !hx.is_finite()||!hy.is_finite()||hx<=0.0||hy<=0.0 {
        return Err("Invalid pixel widths".into());
    }
    if weight_sum==0.0 {
        return Ok(Some(StableFftCertificate{
            width_x:2,width_y:2,pixel_error:0.0,vector_l1_error:0.0
        }));
    }
    let mut best:Option<(usize,usize,StableFftCertificate)>=None;
    for wx in (2..=MAX_STENCIL_WIDTH).step_by(2) {
        let (ex,bx)=axis_errors(wx/2,hx,sx);
        for wy in (2..=MAX_STENCIL_WIDTH).step_by(2) {
            if fft_shape(nx,ny,wx,wy).is_none(){continue;}
            let (ey,by)=axis_errors(wy/2,hy,sy);
            let bound=weight_sum*(ex+ey+ex*ey);
            if bound<=epsilon {
                let cost=wx*wy;
                let tie=wx.max(wy);
                if best.as_ref().is_none_or(|(c,t,_)|
                    cost<*c||(cost==*c&&tie<*t)){
                    best=Some((cost,tie,StableFftCertificate{
                        width_x:wx,width_y:wy,pixel_error:bound,
                        vector_l1_error:weight_sum*(bx+by+bx*by)
                    }));
                }
            }
        }
    }
    Ok(best.map(|(_,_,v)|v))
}

fn coefficients(width:usize)->Vec<f64> {
    let mut out=Vec::with_capacity(width);
    let mut binomial=1.0;
    for i in 0..width {
        out.push(if i%2==0 {binomial} else {-binomial});
        if i+1<width {
            binomial*=((width-1-i) as f64)/((i+1) as f64);
        }
    }
    out
}

fn weights(q:f64,width:usize,signs:&[f64],out:&mut [f64])->isize {
    let start=(q-0.5*(width-1) as f64+0.5).floor() as isize;
    for i in 0..width {
        if q==(start+i as isize) as f64 {
            out.fill(0.0);
            out[i]=1.0;
            return start;
        }
    }
    let mut denominator=0.0;
    for i in 0..width {
        out[i]=signs[i]/(q-(start+i as isize) as f64);
        denominator+=out[i];
    }
    for v in out.iter_mut() { *v/=denominator; }
    start
}

fn fft2(data:&mut [Complex<f64>],lx:usize,ly:usize,
        row:&dyn Fft<f64>,col:&dyn Fft<f64>) {
    for chunk in data.chunks_exact_mut(ly) { row.process(chunk); }
    let mut buf=vec![Complex::new(0.0,0.0);lx];
    for j in 0..ly {
        for i in 0..lx {buf[i]=data[i*ly+j];}
        col.process(&mut buf);
        for i in 0..lx {data[i*ly+j]=buf[i];}
    }
}

/// FFT plan and Gaussian spectrum reusable across diagrams with identical
/// bounds, resolution, bandwidth and interpolation stencil.
pub struct StableIntegratedFft {
    bounds:Bounds, nx:usize,ny:usize,
    hx:f64,hy:f64,sx:f64,sy:f64,weight_scale:f64,
    wx:usize,wy:usize,px:usize,py:usize,lx:usize,ly:usize,
    spectrum:Vec<Complex<f64>>,
    row_forward:Arc<dyn Fft<f64>>,col_forward:Arc<dyn Fft<f64>>,
    row_inverse:Arc<dyn Fft<f64>>,col_inverse:Arc<dyn Fft<f64>>,
    signs_x:Vec<f64>,signs_y:Vec<f64>,
}
impl StableIntegratedFft {
    pub fn new(bounds:Bounds,nx:usize,ny:usize,sx:f64,sy:f64,
               certificate:StableFftCertificate,weight_scale:f64)->Result<Self,String> {
        if !weight_scale.is_finite()||weight_scale<=0.0 {
            return Err("weight_scale must be finite and positive".into());
        }
        let wx=certificate.width_x; let wy=certificate.width_y;
        if wx<2 || wy<2 || wx>MAX_STENCIL_WIDTH || wy>MAX_STENCIL_WIDTH
           || wx%2!=0 || wy%2!=0 {
            return Err("Invalid interpolation stencil width".into());
        }
        let (lx,ly)=fft_shape(nx,ny,wx,wy)
            .ok_or("FFT workspace exceeds allocation limit")?;
        let hx=(bounds.x_max-bounds.x_min)/nx as f64;
        let hy=(bounds.y_max-bounds.y_min)/ny as f64;
        if !(hx.is_finite()&&hx>0.0&&hy.is_finite()&&hy>0.0
             &&sx.is_finite()&&sx>0.0&&sy.is_finite()&&sy>0.0) {
            return Err("Interpolation FFT needs nondegenerate bounds and bandwidths".into());
        }
        let px=(wx+2).max(3); let py=(wy+2).max(3);
        let mut planner=FftPlanner::<f64>::new();
        let row_forward=planner.plan_fft_forward(ly);
        let col_forward=planner.plan_fft_forward(lx);
        let row_inverse=planner.plan_fft_inverse(ly);
        let col_inverse=planner.plan_fft_inverse(lx);
        // Probability in the pixel centered at lattice offset d*h, NOT
        // the Gaussian density sampled at that center.
        let gx:Vec<f64>=(0..lx).map(|i|{
            let d=i.min(lx-i) as f64;
            cdf_interval((d-0.5)*hx,(d+0.5)*hx,0.0,sx)
        }).collect();
        let gy:Vec<f64>=(0..ly).map(|j|{
            let d=j.min(ly-j) as f64;
            cdf_interval((d-0.5)*hy,(d+0.5)*hy,0.0,sy)
        }).collect();
        let mut spectrum=vec![Complex::new(0.0,0.0);lx*ly];
        for i in 0..lx {
            for j in 0..ly {spectrum[i*ly+j].re=gx[i]*gy[j];}
        }
        fft2(&mut spectrum,lx,ly,row_forward.as_ref(),col_forward.as_ref());
        Ok(Self{bounds,nx,ny,hx,hy,sx,sy,weight_scale,wx,wy,px,py,lx,ly,spectrum,
                row_forward,col_forward,row_inverse,col_inverse,
                signs_x:coefficients(wx),signs_y:coefficients(wy)})
    }

    /// Returns x-major row-major pixels: image[i*ny+j].
    pub fn evaluate(&self,diagram:&Diagram)->Result<Vec<f64>,String>{
        if diagram.is_empty(){return Ok(vec![0.0;self.nx*self.ny]);}
        let mut mass=vec![Complex::new(0.0,0.0);self.lx*self.ly];
        let mut ax=vec![0.0;self.wx];
        let mut ay=vec![0.0;self.wy];
        let mx=self.nx+2*self.px;
        let my=self.ny+2*self.py;
        for &(b,p) in diagram {
            if !b.is_finite()||!p.is_finite()
                ||b<self.bounds.x_min||b>self.bounds.x_max
                ||p<self.bounds.y_min||p>self.bounds.y_max {
                return Err("Interpolation FFT needs finite sources inside bounds".into());
            }
            if p<0.0 {return Err("Stable FFT requires nonnegative persistence".into());}
            let weight=(p/self.weight_scale).min(1.0);
            if weight==0.0 {continue;}
            let qx=(b-(self.bounds.x_min+0.5*self.hx))/self.hx;
            let qy=(p-(self.bounds.y_min+0.5*self.hy))/self.hy;
            let ix=weights(qx,self.wx,&self.signs_x,&mut ax);
            let iy=weights(qy,self.wy,&self.signs_y,&mut ay);
            for (i,&a) in ax.iter().enumerate() {
                let row=ix+self.px as isize+i as isize;
                if row<0||row>=mx as isize{return Err("x ghost padding exhausted".into());}
                for (j,&v) in ay.iter().enumerate() {
                    let col=iy+self.py as isize+j as isize;
                    if col<0||col>=my as isize{return Err("y ghost padding exhausted".into());}
                    mass[row as usize*self.ly+col as usize].re+=weight*a*v;
                }
            }
        }
        fft2(&mut mass,self.lx,self.ly,self.row_forward.as_ref(),self.col_forward.as_ref());
        for (a,&b) in mass.iter_mut().zip(&self.spectrum) {*a*=b;}
        fft2(&mut mass,self.lx,self.ly,self.row_inverse.as_ref(),self.col_inverse.as_ref());
        let scale=1.0/(self.lx as f64*self.ly as f64);
        let mut result=vec![0.0;self.nx*self.ny];
        for i in 0..self.nx {
            for j in 0..self.ny {
                result[i*self.ny+j]=(mass[(i+self.px)*self.ly+(j+self.py)].re*scale).max(0.0);
            }
        }
        Ok(result)
    }

    pub fn stencil(&self)->(usize,usize){(self.wx,self.wy)}
    pub fn bandwidths(&self)->(f64,f64){(self.sx,self.sy)}
}

/// Explicit FFT approximation with both interpolation certificates.
/// Sources must be inside the supplied fixed bounds. The error bounds do
/// not account for roundoff during signed deposition, FFT or clipping.
pub fn stable_interp_fft_image_with_certificate(
    diagram:&Diagram,bounds:Bounds,nx:usize,ny:usize,sx:f64,sy:f64,
    weight_scale:f64,epsilon:f64)->Result<(Vec<f64>,StableFftCertificate),String>{
    if !weight_scale.is_finite()||weight_scale<=0.0 {
        return Err("weight_scale must be finite and positive".into());
    }
    if diagram.iter().any(|&(b,p)|!b.is_finite()||!p.is_finite()||p<0.0) {
        return Err("Stable FFT requires finite birth and nonnegative persistence".into());
    }
    let weight_sum:f64=diagram.iter().map(|&(_,p)|(p/weight_scale).min(1.0)).sum();
    let cert=select_stable_fft_stencil(weight_sum,bounds,nx,ny,sx,sy,epsilon)?
        .ok_or("No integrated FFT stencil certifies absolute_error")?;
    let image=if weight_sum==0.0 {
        vec![0.0;nx*ny]
    }else{
        StableIntegratedFft::new(bounds,nx,ny,sx,sy,cert,weight_scale)?.evaluate(diagram)?
    };
    Ok((image,cert))
}

/// Explicit approximate stable image, requiring a positive error budget.
pub fn stable_interp_fft_image(
    diagram:&Diagram,bounds:Bounds,nx:usize,ny:usize,sx:f64,sy:f64,
    weight_scale:f64,epsilon:f64)->Result<Vec<f64>,String>{
    Ok(stable_interp_fft_image_with_certificate(
        diagram,bounds,nx,ny,sx,sy,weight_scale,epsilon)?.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stable::stable_persistence_image;
    fn bounds()->Bounds {Bounds{x_min:0.0,x_max:1.0,y_min:0.0,y_max:1.0}}
    fn max_diff(a:&[f64],b:&[f64])->f64{
        a.iter().zip(b).map(|(x,y)|(x-y).abs()).fold(0.0_f64,f64::max)
    }
    #[test]
    fn anisotropy_boundaries_and_error_certificates(){
        let b=Bounds{x_min:-1.0,x_max:2.0,y_min:0.0,y_max:3.0};
        let d=vec![(-1.0,0.0),(0.1,0.4),(2.0,3.0),(-0.6,2.4)];
        let (out,cert)=stable_interp_fft_image_with_certificate(
            &d,b,80,56,0.37,0.51,0.8,1e-8).unwrap();
        let reference=stable_persistence_image(&d,b,80,56,0.37,0.51,0.8).unwrap();
        let err=max_diff(&out,&reference);
        let l1:f64=out.iter().zip(&reference).map(|(a,b)|(a-b).abs()).sum();
        assert!(err<=cert.pixel_error+2e-11,"observed {err}, bound {}",cert.pixel_error);
        assert!(l1<=cert.vector_l1_error+2e-10,"L1 observed {l1}, bound {}",cert.vector_l1_error);
        assert!(out.iter().all(|&x|x>=0.0));
    }
    #[test]
    fn continuous_and_repeated_points(){
        let d:Diagram=(0..500).map(|k|(
            (k*37%503) as f64/503.0,
            (k*113%509) as f64/509.0
        )).collect();
        let (img,cert)=stable_interp_fft_image_with_certificate(
            &d,bounds(),96,80,0.18,0.21,0.6,1e-8).unwrap();
        let ref_=stable_persistence_image(&d,bounds(),96,80,0.18,0.21,0.6).unwrap();
        assert!(max_diff(&img,&ref_)<=cert.pixel_error+5e-11);
    }
    #[test]
    fn zero_weight_and_rejections(){
        let zero=vec![(0.0,0.0);20];
        let (image,cert)=stable_interp_fft_image_with_certificate(
            &zero,bounds(),64,48,.15,.2,1.0,1e-8).unwrap();
        assert!(image.iter().all(|&v|v==0.0));
        assert_eq!(cert.pixel_error,0.0);
        assert!(stable_interp_fft_image(&vec![(0.5,0.5)],bounds(),16,16,
            0.0001,0.0001,1.0,1e-10).is_err());
        assert!(stable_interp_fft_image(&vec![(0.5,0.5)],bounds(),64,64,
            .2,.2,1.0,0.0).is_err());
        assert!(stable_interp_fft_image(&vec![(0.5,-0.2)],bounds(),64,64,
            .2,.2,1.0,1e-8).is_err());
        assert!(stable_interp_fft_image(&vec![(2.5,0.2)],bounds(),64,64,
            .2,.2,1.0,1e-8).is_err());
    }
}
