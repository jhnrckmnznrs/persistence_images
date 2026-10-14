//! Error-controlled one-FFT persistence images using tensor Lagrange source gridding.
//!
//! The certificate controls interpolation truncation in exact arithmetic, not
//! floating-point deposition, cancellation, or FFT roundoff. The lattice is
//! extended with ghost nodes so input points on the domain boundary are valid.
use crate::diagram::Diagram;
use crate::image::Bounds;
use rustfft::{Fft, FftPlanner, num_complex::Complex};
use std::sync::Arc;

const MAX_STENCIL_WIDTH: usize = 18;
const MAX_FFT_CELLS: usize = 4_000_000;

#[derive(Debug, Clone, Copy)]
pub struct StencilCertificate {
    pub width_x: usize,
    pub width_y: usize,
    /// Uniform absolute *interpolation* error bound per image pixel.
    pub error_bound: f64,
}

fn log_factorial(n: usize) -> f64 {
    (2..=n).map(|i| (i as f64).ln()).sum()
}

/// One-source, one-dimensional upper bound for a central stencil of 2r nodes.
fn axis_error(r: usize, h: f64, sigma: f64) -> f64 {
    if h == 0.0 { return 0.0; }
    let t = h / sigma;
    if !t.is_finite() || t <= 0.0 { return f64::INFINITY; }
    let r_f = r as f64;
    let log_e = 2.0*log_factorial(2*r) - 3.0*log_factorial(r)
        - 5.0*r_f*std::f64::consts::LN_2 + 2.0*r_f*t.ln();
    if log_e > 700.0 { f64::INFINITY } else { log_e.exp() }
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

/// Select the cheapest (by deposit width product) certified even stencil.
/// Returns None if the grid is degenerate, too large, or no supported
/// stencil achieves the requested tolerance.
pub fn select_stencil(n:usize, bounds:Bounds, nx:usize,ny:usize,
                      sigma_x:f64,sigma_y:f64,epsilon:f64)
                      ->Result<Option<StencilCertificate>,String> {
    if nx==0 || ny==0 || nx.checked_mul(ny).is_none() {
        return Err("Grid sizes must be positive and representable".into());
    }
    if !epsilon.is_finite() || epsilon<=0.0 {
        return Err("absolute_error must be finite and positive".into());
    }
    if !(sigma_x.is_finite() && sigma_x>0.0 && sigma_y.is_finite() && sigma_y>0.0)
        || !(bounds.x_min.is_finite() && bounds.x_max.is_finite()
           && bounds.y_min.is_finite() && bounds.y_max.is_finite())
        || bounds.x_min>bounds.x_max || bounds.y_min>bounds.y_max {
        return Err("Invalid bounds or Gaussian bandwidths".into());
    }
    let hx=(bounds.x_max-bounds.x_min)/nx as f64;
    let hy=(bounds.y_max-bounds.y_min)/ny as f64;
    if !(hx.is_finite() && hy.is_finite()) || hx<=0.0 || hy<=0.0 {
        return Ok(None);
    }
    let mut best:Option<(usize,StencilCertificate)>=None;
    for wx in (2..=MAX_STENCIL_WIDTH).step_by(2) {
        let ex=axis_error(wx/2,hx,sigma_x);
        for wy in (2..=MAX_STENCIL_WIDTH).step_by(2) {
            if fft_shape(nx,ny,wx,wy).is_none() {continue;}
            let ey=axis_error(wy/2,hy,sigma_y);
            let bound=(n as f64)*(ex+ey+ex*ey);
            if bound<=epsilon {
                let cost=wx*wy;
                if best.as_ref().is_none_or(|(c,_)|cost<*c) {
                    best=Some((cost,StencilCertificate{
                        width_x:wx,width_y:wy,error_bound:bound
                    }));
                }
            }
        }
    }
    Ok(best.map(|(_,c)|c))
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
pub struct InterpFft {
    bounds:Bounds, nx:usize,ny:usize,
    hx:f64,hy:f64,sx:f64,sy:f64,
    wx:usize,wy:usize,px:usize,py:usize,lx:usize,ly:usize,
    spectrum:Vec<Complex<f64>>,
    row_forward:Arc<dyn Fft<f64>>,col_forward:Arc<dyn Fft<f64>>,
    row_inverse:Arc<dyn Fft<f64>>,col_inverse:Arc<dyn Fft<f64>>,
    signs_x:Vec<f64>,signs_y:Vec<f64>,
}
impl InterpFft {
    pub fn new(bounds:Bounds,nx:usize,ny:usize,sx:f64,sy:f64,
               certificate:StencilCertificate)->Result<Self,String> {
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
        let gx:Vec<f64>=(0..lx).map(|i|{
            let t=(i.min(lx-i) as f64)*(hx/sx);
            (-0.5*t*t).exp()
        }).collect();
        let gy:Vec<f64>=(0..ly).map(|j|{
            let t=(j.min(ly-j) as f64)*(hy/sy);
            (-0.5*t*t).exp()
        }).collect();
        let mut spectrum=vec![Complex::new(0.0,0.0);lx*ly];
        for i in 0..lx {
            for j in 0..ly {spectrum[i*ly+j].re=gx[i]*gy[j];}
        }
        fft2(&mut spectrum,lx,ly,row_forward.as_ref(),col_forward.as_ref());
        Ok(Self{bounds,nx,ny,hx,hy,sx,sy,wx,wy,px,py,lx,ly,spectrum,
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
                    mass[row as usize*self.ly+col as usize].re+=a*v;
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
                result[i*self.ny+j]=mass[(i+self.px)*self.ly+(j+self.py)].re*scale;
            }
        }
        Ok(result)
    }

    pub fn stencil(&self)->(usize,usize){(self.wx,self.wy)}
    pub fn bandwidths(&self)->(f64,f64){(self.sx,self.sy)}
}

/// Convenience API: explicit tolerance required; errors if no supported
/// interpolation stencil can certify it.
pub fn interp_fft_image(diagram:&Diagram,bounds:Bounds,nx:usize,ny:usize,
                        sx:f64,sy:f64,epsilon:f64)->Result<Vec<f64>,String> {
    if diagram.is_empty() {return Ok(vec![0.0;nx.checked_mul(ny).ok_or("Invalid grid size")?]);}
    let cert=select_stencil(diagram.len(),bounds,nx,ny,sx,sy,epsilon)?
        .ok_or("No FFT interpolation stencil certifies this error tolerance")?;
    InterpFft::new(bounds,nx,ny,sx,sy,cert)?.evaluate(diagram)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::persistence_image;
    fn bounds()->Bounds { Bounds{x_min:0.0,x_max:1.0,y_min:0.0,y_max:1.0} }
    fn assert_close(a:&[f64],b:&[f64],tol:f64){
        assert_eq!(a.len(),b.len());
        let diff=a.iter().zip(b).map(|(a,b)|(a-b).abs()).fold(0.0_f64,f64::max);
        assert!(diff<=tol,"max error {diff} exceeded {tol}");
    }
    #[test]
    fn stencil_errors_and_shape() {
        let c=select_stencil(100,bounds(),64,40,0.35,0.25,1e-8)
            .unwrap().unwrap();
        assert!(c.error_bound<=1e-8);
        assert_eq!(c.width_x%2,0);
        let (lx,ly)=fft_shape(64,40,c.width_x,c.width_y).unwrap();
        assert!(lx>=128 && ly>=80);
        assert!(select_stencil(100,bounds(),12,12,0.00001,0.00001,1e-10).unwrap().is_none());
    }
    #[test]
    fn boundaries_anisotropy_and_order() {
        let b=Bounds{x_min:-2.0,x_max:3.0,y_min:4.0,y_max:8.0};
        let d:Diagram=vec![(-2.0,4.0),(3.0,8.0),(0.25,5.0),
                           (-0.25,7.8),(-2.0,8.0)];
        let c=select_stencil(d.len(),b,72,53,0.9,0.65,1e-8)
            .unwrap().unwrap();
        let engine=InterpFft::new(b,72,53,0.9,0.65,c).unwrap();
        let out=engine.evaluate(&d).unwrap();
        let reference=persistence_image(&d,b,72,53,0.9,0.65);
        assert_close(&out,&reference,1e-8);
        assert_eq!(out.len(),72*53);
        assert_eq!(engine.stencil(),(c.width_x,c.width_y));
    }
    #[test]
    fn repeated_and_empty_diagrams() {
        let d:Diagram=(0..180).map(|k|
            (((k*17 % 173) as f64)/173.0,((k*31%179) as f64)/179.0)).collect();
        let reference=persistence_image(&d,bounds(),64,48,0.2,0.35);
        let approximation=interp_fft_image(&d,bounds(),64,48,0.2,0.35,1e-7).unwrap();
        assert_close(&approximation,&reference,1e-7);
        let none=interp_fft_image(&vec![],bounds(),64,48,0.2,0.35,1e-8).unwrap();
        assert_eq!(none,vec![0.0;64*48]);
    }
    #[test]
    fn narrow_kernel_rejected() {
        let d=vec![(0.51,0.49);25];
        assert!(interp_fft_image(&d,bounds(),12,12,0.0001,0.0001,1e-10).is_err());
    }
}
