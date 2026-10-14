//! Adaptive persistence-image computation. All coordinates are (birth,persistence).
//! The original image::persistence_image remains the direct numerical reference.
use crate::diagram::Diagram;
use crate::image::{Bounds, persistence_image};
use crate::interp_fft::{InterpFft, select_stencil, interp_fft_image, fft_shape};
use clap::ValueEnum;
use rayon::prelude::*;
use std::collections::BTreeMap;

const MAX_CELLS: usize = 2_000_000;
const MAX_BASIS: usize = 16_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Engine { Auto, Direct, Separable, Shared, Sparse, InterpFft }

fn check(bounds: Bounds, nx: usize, ny: usize, sx: f64, sy: f64) -> Result<(), String> {
    if nx == 0 || ny == 0 || nx.checked_mul(ny).is_none() {
        return Err("nx and ny must be positive with a representable product".into());
    }
    if !(sx.is_finite() && sx > 0.0 && sy.is_finite() && sy > 0.0) {
        return Err("Gaussian bandwidths must be finite and positive".into());
    }
    if !(bounds.x_min.is_finite() && bounds.x_max.is_finite()
        && bounds.y_min.is_finite() && bounds.y_max.is_finite()
        && bounds.x_min <= bounds.x_max && bounds.y_min <= bounds.y_max) {
        return Err("Invalid global bounds".into());
    }
    Ok(())
}
fn check_diagram(diagram: &Diagram) -> Result<(), String> {
    if diagram.iter().any(|&(b,p)| !b.is_finite() || !p.is_finite()) {
        return Err("Diagram coordinates must be finite".into());
    }
    Ok(())
}
fn centers(bounds: Bounds, nx: usize, ny: usize) -> (Vec<f64>, Vec<f64>) {
    let dx=(bounds.x_max-bounds.x_min)/nx as f64;
    let dy=(bounds.y_max-bounds.y_min)/ny as f64;
    ((0..nx).map(|i| bounds.x_min+(i as f64+0.5)*dx).collect(),
     (0..ny).map(|j| bounds.y_min+(j as f64+0.5)*dy).collect())
}
fn separable_kernel(dgm: &Diagram, xs: &[f64], ys: &[f64], sx: f64, sy: f64) -> Vec<f64> {
    let (nx,ny)=(xs.len(),ys.len());
    let mut out=vec![0.0;nx*ny];
    let (mut gx, mut gy)=(vec![0.0;nx],vec![0.0;ny]);
    for &(b,p) in dgm {
        for (i,&x) in xs.iter().enumerate() { let z=(x-b)/sx; gx[i]=(-0.5*z*z).exp(); }
        for (j,&y) in ys.iter().enumerate() { let z=(y-p)/sy; gy[j]=(-0.5*z*z).exp(); }
        for i in 0..nx { for j in 0..ny { out[i*ny+j]+=gx[i]*gy[j]; } }
    }
    out
}
pub fn separable_image(dgm: &Diagram, bounds: Bounds, nx: usize, ny: usize,
                       sx: f64, sy: f64) -> Result<Vec<f64>,String> {
    check(bounds,nx,ny,sx,sy)?;
    check_diagram(dgm)?;
    let (xs,ys)=centers(bounds,nx,ny);
    Ok(separable_kernel(dgm,&xs,&ys,sx,sy))
}
fn cutoff(n: usize, epsilon: f64) -> Result<f64,String> {
    if !epsilon.is_finite() || epsilon <= 0.0 { return Err("absolute_error must be finite and positive".into()); }
    // Splitting logarithms avoids overflow in N / epsilon.
    Ok((2.0*((n.max(1) as f64).ln()-epsilon.ln()).max(0.0)).sqrt())
}
fn window(xs: &[f64],ys: &[f64],b:f64,p:f64,wx:f64,wy:f64) -> (usize,usize,usize,usize) {
    (xs.partition_point(|&v|v<b-wx),xs.partition_point(|&v|v<=b+wx),
     ys.partition_point(|&v|v<p-wy),ys.partition_point(|&v|v<=p+wy))
}
/// Per-pixel absolute truncation error <= epsilon in exact arithmetic,
/// because each omitted Gaussian is <= exp(-r²/2) and r=sqrt(2log(N/epsilon)).
/// Floating-point roundoff is not included in this guarantee.
pub fn sparse_image(dgm: &Diagram,bounds:Bounds,nx:usize,ny:usize,
                    sx:f64,sy:f64,epsilon:f64) -> Result<Vec<f64>,String> {
    check(bounds,nx,ny,sx,sy)?;
    check_diagram(dgm)?;
    let r=cutoff(dgm.len(),epsilon)?;
    let (xs,ys)=centers(bounds,nx,ny);
    let mut out=vec![0.0;nx*ny];
    for &(b,p) in dgm {
        let (i0,i1,j0,j1)=window(&xs,&ys,b,p,r*sx,r*sy);
        for i in i0..i1 {
            let z=(xs[i]-b)/sx;
            let gx=(-0.5*z*z).exp();
            for j in j0..j1 {
                let z=(ys[j]-p)/sy;
                out[i*ny+j]+=gx*(-0.5*z*z).exp();
            }
        }
    }
    Ok(out)
}
pub fn adaptive_image(dgm:&Diagram,bounds:Bounds,nx:usize,ny:usize,
                      sx:f64,sy:f64,engine:Engine,epsilon:Option<f64>)
                      -> Result<Vec<f64>,String> {
    check(bounds,nx,ny,sx,sy)?;
    check_diagram(dgm)?;
    let radius=epsilon.map(|e|cutoff(dgm.len(),e)).transpose()?;
    match engine {
        Engine::Direct=>Ok(persistence_image(dgm,bounds,nx,ny,sx,sy)),
        Engine::Separable|Engine::Shared=>separable_image(dgm,bounds,nx,ny,sx,sy),
        Engine::Sparse=>sparse_image(dgm,bounds,nx,ny,sx,sy,
                                    epsilon.ok_or("sparse requires --absolute-error")?),
        Engine::InterpFft=>interp_fft_image(dgm,bounds,nx,ny,sx,sy,
                                    epsilon.ok_or("interp-fft requires --absolute-error")?),
        Engine::Auto=>{
            if let Some(r)=radius {
                if dgm.len()>=128 && nx*ny>=1024 {
                    let (xs,ys)=centers(bounds,nx,ny);
                    let mut local=0.0;
                    for &(b,p) in dgm {
                        let (i0,i1,j0,j1)=window(&xs,&ys,b,p,r*sx,r*sy);
                        local+=((i1-i0) as f64)*((j1-j0) as f64);
                    }
                    if local < 0.12*(dgm.len() as f64)*(nx as f64)*(ny as f64) {
                        return sparse_image(dgm,bounds,nx,ny,sx,sy,
                                            epsilon.expect("validated tolerance"));
                    }
                }
            }
            if let Some(e)=epsilon {
                if dgm.len()>=4096 && nx*ny>=4096 {
                    if let Some(c)=select_stencil(dgm.len(),bounds,nx,ny,sx,sy,e)? {
                        if dgm.iter().all(|&(b,p)| b>=bounds.x_min && b<=bounds.x_max
                            && p>=bounds.y_min && p<=bounds.y_max) {
                            if let Some((lx,ly))=fft_shape(nx,ny,c.width_x,c.width_y) {
                                let n=dgm.len() as f64;
                                let work=n*(c.width_x*c.width_y) as f64
                                    + 30.0*(lx*ly) as f64*((lx*ly) as f64).log2();
                                if n*(nx*ny) as f64 > 8.0*work {
                                    return InterpFft::new(bounds,nx,ny,sx,sy,c)?.evaluate(dgm);
                                }
                            }
                        }
                    }
                }
            }
            separable_image(dgm,bounds,nx,ny,sx,sy)
        }
    }
}
fn unique(mut v:Vec<f64>)->Vec<f64> {
    v.sort_by(f64::total_cmp);
    v.dedup_by(|a,b|a.total_cmp(b).is_eq());
    v
}
/// Cached A[x,b] and D[p,y] for diagrams with the same grid and bandwidth.
/// Each diagram supplies its own coordinate multiplicity matrix C[b,p].
#[derive(Debug)]
pub struct SharedGaussianBasis {
    births:Vec<f64>, pers:Vec<f64>,
    a:Vec<f64>,d:Vec<f64>,nx:usize,ny:usize,
}
impl SharedGaussianBasis {
    pub fn from_diagrams(diagrams:&[&Diagram],bounds:Bounds,nx:usize,ny:usize,
                         sx:f64,sy:f64)->Result<Self,String> {
        check(bounds,nx,ny,sx,sy)?;
        for d in diagrams { check_diagram(d)?; }
        let births=unique(diagrams.iter().flat_map(|d|d.iter().map(|p|p.0)).collect());
        let pers=unique(diagrams.iter().flat_map(|d|d.iter().map(|p|p.1)).collect());
        let (nb,np)=(births.len(),pers.len());
        if nb.checked_mul(np).is_none_or(|n|n>MAX_CELLS)
            || nx.checked_mul(nb).and_then(|a|ny.checked_mul(np).and_then(|b|a.checked_add(b)))
                .is_none_or(|n|n>MAX_BASIS) {
            return Err("Shared Gaussian basis exceeds allocation limits".into());
        }
        let (xs,ys)=centers(bounds,nx,ny);
        let mut a=Vec::with_capacity(nx*nb);
        for &x in &xs { for &b in &births {
            let z=(x-b)/sx; a.push((-0.5*z*z).exp());
        }}
        let mut d=Vec::with_capacity(np*ny);
        for &p in &pers { for &y in &ys {
            let z=(y-p)/sy; d.push((-0.5*z*z).exp());
        }}
        Ok(Self{births,pers,a,d,nx,ny})
    }
    pub fn coordinate_counts(&self)->(usize,usize) { (self.births.len(),self.pers.len()) }
    pub fn transform(&self,diagram:&Diagram)->Result<Vec<f64>,String> {
        let (nb,np)=self.coordinate_counts();
        if nb==0 || np==0 {
            return if diagram.is_empty(){Ok(vec![0.0;self.nx*self.ny])}
                   else{Err("Nonempty diagram with empty basis".into())};
        }
        let mut counts=vec![0.0_f64;nb*np];
        for &(b,p) in diagram {
            let u=self.births.binary_search_by(|v|v.total_cmp(&b))
                .map_err(|_|"Birth not in shared basis")?;
            let v=self.pers.binary_search_by(|v|v.total_cmp(&p))
                .map_err(|_|"Persistence not in shared basis")?;
            counts[u*np+v]+=1.0;
        }
        let (nx,ny)=(self.nx,self.ny);
        let mut out=vec![0.0;nx*ny];
        let lc=(nx as u128)*(nb as u128)*(np as u128)
              +(nx as u128)*(np as u128)*(ny as u128);
        let rc=(nb as u128)*(np as u128)*(ny as u128)
              +(nx as u128)*(nb as u128)*(ny as u128);
        if lc<=rc {
            let mut tmp=vec![0.0;nx*np];
            for u in 0..nb {for v in 0..np {
                let c=counts[u*np+v]; if c==0.0 {continue;}
                for i in 0..nx {tmp[i*np+v]+=c*self.a[i*nb+u];}
            }}
            for i in 0..nx {for v in 0..np {
                let t=tmp[i*np+v]; if t==0.0 {continue;}
                for j in 0..ny {out[i*ny+j]+=t*self.d[v*ny+j];}
            }}
        } else {
            let mut tmp=vec![0.0;nb*ny];
            for u in 0..nb {for v in 0..np {
                let c=counts[u*np+v]; if c==0.0 {continue;}
                for j in 0..ny {tmp[u*ny+j]+=c*self.d[v*ny+j];}
            }}
            for i in 0..nx {for u in 0..nb {
                let a=self.a[i*nb+u];
                for j in 0..ny {out[i*ny+j]+=a*tmp[u*ny+j];}
            }}
        }
        Ok(out)
    }
}
fn shared_candidate(ds:&[&Diagram],nx:usize,ny:usize)->bool {
    if ds.len()<2||nx*ny<256{return false;}
    let n:usize=ds.iter().map(|d|d.len()).sum();
    if n<512{return false;}
    let sample:Vec<_>=ds.iter().flat_map(|d|d.iter().take(512).copied()).take(2048).collect();
    if sample.is_empty(){return false;}
    let nb=unique(sample.iter().map(|p|p.0).collect()).len();
    let np=unique(sample.iter().map(|p|p.1).collect()).len();
    nb.saturating_mul(np)<=8*sample.len()
}
fn shared_economical(b:&SharedGaussianBasis,ds:&[&Diagram],nx:usize,ny:usize)->bool {
    let (nb,np)=b.coordinate_counts();
    let n:u128=ds.iter().map(|d|d.len() as u128).sum();
    let (x,y,bu,pu,g)=(nx as u128,ny as u128,nb as u128,np as u128,ds.len() as u128);
    let shared=12*(x*bu+y*pu)+8*n+g*(bu*pu*x.min(y)+x*y*bu.min(pu));
    let dense=n*(x*y+4*(x+y));
    10*shared < 6*dense
}
/// Batch evaluator. Input diagrams are already spacing-scaled. The spacings
/// argument scales Gaussian bandwidths, just as in the original Rust CLI.
/// Auto is mathematically exact unless absolute_error is explicitly supplied.
pub fn batch_images(diagrams:&[Diagram],spacings:&[f64],bounds:Bounds,
                    nx:usize,ny:usize,sx:f64,sy:f64,engine:Engine,
                    absolute_error:Option<f64>)->Result<Vec<Vec<f64>>,String> {
    check(bounds,nx,ny,sx,sy)?;
    if diagrams.len()!=spacings.len(){return Err("spacings length mismatch".into());}
    if let Some(e)=absolute_error{cutoff(1,e)?;}
    for d in diagrams{check_diagram(d)?;}
    for &s in spacings{
        if !s.is_finite()||s<=0.0||(sx*s)<=0.0||(sy*s)<=0.0
            ||!(sx*s).is_finite()||!(sy*s).is_finite(){
            return Err("Scaled bandwidths must be finite and positive".into());
        }
    }
    let mut groups=BTreeMap::<u64,Vec<usize>>::new();
    for (i,s) in spacings.iter().enumerate(){groups.entry(s.to_bits()).or_default().push(i);}
    let mut out=vec![Vec::new();diagrams.len()];
    for idx in groups.values(){
        let ds:Vec<&Diagram>=idx.iter().map(|&i|&diagrams[i]).collect();
        let scale=spacings[idx[0]];
        let local_sx=sx*scale;let local_sy=sy*scale;
        if engine==Engine::Shared||(engine==Engine::Auto&&shared_candidate(&ds,nx,ny)){
            match SharedGaussianBasis::from_diagrams(&ds,bounds,nx,ny,local_sx,local_sy){
                Ok(basis)=>{
                    if engine==Engine::Shared||shared_economical(&basis,&ds,nx,ny){
                        for &i in idx{out[i]=basis.transform(&diagrams[i])?;}
                        continue;
                    }
                },
                Err(e) if engine==Engine::Shared=>return Err(e),
                Err(_)=>(),
            }
        }
        // A single Gaussian spectrum and set of FFT plans can be reused
        // across this complete group, because spacings and bounds match.
        if (engine==Engine::InterpFft || engine==Engine::Auto)
           && absolute_error.is_some() {
            let e=absolute_error.expect("validated");
            let nmax=ds.iter().map(|d|d.len()).max().unwrap_or(0);
            if nmax==0 {
                for &i in idx {out[i]=vec![0.0;nx*ny];}
                continue;
            }
            let selected=select_stencil(nmax,bounds,nx,ny,local_sx,local_sy,e)?;
            if engine==Engine::InterpFft && selected.is_none() {
                return Err("No FFT interpolation stencil certifies this error tolerance".into());
            }
            if let Some(c)=selected {
                let inside=ds.iter().all(|d|d.iter().all(|&(b,p)|
                    b>=bounds.x_min&&b<=bounds.x_max&&p>=bounds.y_min&&p<=bounds.y_max));
                if engine==Engine::InterpFft && !inside {
                    return Err("Interpolation FFT needs source coordinates inside bounds".into());
                }
                let use_fft=engine==Engine::InterpFft || (inside
                    && nx*ny>=4096 && idx.iter().any(|&i|diagrams[i].len()>=4096)
                    && fft_shape(nx,ny,c.width_x,c.width_y).is_some_and(|(lx,ly)|{
                        let n:usize=idx.iter().map(|&i|diagrams[i].len()).sum();
                        let work=n as f64*(c.width_x*c.width_y) as f64
                            + 30.0*(lx*ly) as f64*((lx*ly) as f64).log2();
                        (n as f64)*(nx*ny) as f64 > 8.0*work
                    }));
                if use_fft {
                    let plan=InterpFft::new(bounds,nx,ny,local_sx,local_sy,c)?;
                    for &i in idx {out[i]=plan.evaluate(&diagrams[i])?;}
                    continue;
                }
            }
        }
        let computed:Vec<Result<Vec<f64>,String>>=idx.par_iter().map(|&i|{
            adaptive_image(&diagrams[i],bounds,nx,ny,local_sx,local_sy,engine,absolute_error)
        }).collect();
        for (&i,im) in idx.iter().zip(computed){out[i]=im?;}
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn close(x:&[f64],y:&[f64],tol:f64){
        assert_eq!(x.len(),y.len());
        for (&a,&b) in x.iter().zip(y){assert!((a-b).abs()<=tol*(1.0+a.abs()),"{a} != {b}");}
    }
    #[test]
    fn exact_backends(){
        let b=Bounds{x_min:0.0,x_max:2.0,y_min:0.0,y_max:3.0};
        let a=vec![(0.0,1.0),(1.0,2.0),(0.0,1.0)];
        let c=vec![(1.0,2.0),(1.0,2.0)];
        let basis=SharedGaussianBasis::from_diagrams(&[&a,&c],b,27,25,0.3,0.5).unwrap();
        for d in [&a,&c]{
            let reference=persistence_image(d,b,27,25,0.3,0.5);
            close(&basis.transform(d).unwrap(),&reference,1e-12);
            close(&separable_image(d,b,27,25,0.3,0.5).unwrap(),&reference,1e-12);
        }
        assert!(basis.transform(&vec![(0.33,1.0)]).is_err());
    }
    #[test]
    fn error_bound_degenerate_and_spacing(){
        let b=Bounds{x_min:0.0,x_max:1.0,y_min:0.0,y_max:1.0};
        let d:Diagram=(0..50).map(|i|((i%11) as f64/11.0,(i%13) as f64/13.0)).collect();
        let reference=separable_image(&d,b,32,33,0.015,0.009).unwrap();
        close(&sparse_image(&d,b,32,33,0.015,0.009,1e-8).unwrap(),&reference,1e-8);
        let ds=vec![d.clone(),d.clone()];
        let scaled=vec![1.0,2.0];
        let out=batch_images(&ds,&scaled,b,32,33,0.1,0.2,Engine::Shared,None).unwrap();
        for (i,img) in out.iter().enumerate(){
            close(img,&persistence_image(&d,b,32,33,0.1*scaled[i],0.2*scaled[i]),1e-12);
        }
        let flat=Bounds{x_min:0.2,x_max:0.2,y_min:0.3,y_max:0.3};
        close(&sparse_image(&d,flat,3,4,0.02,0.02,1e-8).unwrap(),
              &separable_image(&d,flat,3,4,0.02,0.02).unwrap(),1e-8);
    }
    #[test]
    fn invalid(){
        let b=Bounds{x_min:0.0,x_max:1.0,y_min:0.0,y_max:1.0};
        assert!(separable_image(&vec![],b,0,2,1.0,1.0).is_err());
        assert!(sparse_image(&vec![],b,2,2,1.0,1.0,0.0).is_err());
        assert!(batch_images(&[vec![]],&[],b,2,2,1.0,1.0,Engine::Auto,None).is_err());
        assert_eq!(separable_image(&vec![],b,2,3,1.0,1.0).unwrap(),vec![0.0;6]);
    }
}
