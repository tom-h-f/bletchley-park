use std::fmt;

/// Fixed-size Matrix with const-generics
#[derive(Clone, Copy, PartialEq)]
pub struct Matrix<T, const R: usize, const C: usize> {
    data: [[T; C]; R],
}

impl<T: Default + Copy, const R: usize, const C: usize> Matrix<T, R, C> {
    pub fn new() -> Self {
        Self {
            data: [[T::default(); C]; R],
        }
    }

    #[inline]
    pub const fn get(&self, i: usize, j: usize) -> &T {
        &self.data[i][j]
    }
}

impl<T: fmt::Display, const R: usize, const C: usize> fmt::Display for Matrix<T, R, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if R == 0 || C == 0 {
            writeln!(f, "{}x{} Matrix: []", R, C)?;
            return Ok(());
        }

        // global max width for uniform columns
        let mut maxw = 0usize;
        for r in 0..R {
            for c in 0..C {
                let s = format!("{}", self.data[r][c]);
                maxw = maxw.max(s.len());
            }
        }

        writeln!(f, "{}x{} Matrix:", R, C)?;
        for r in 0..R {
            write!(f, "[")?;
            for c in 0..C {
                if c > 0 {
                    write!(f, " ")?;
                }
                let s = format!("{}", self.data[r][c]);
                write!(f, "{:>width$}", s, width = maxw)?;
            }
            writeln!(f, "]")?;
        }
        Ok(())
    }
}

// Diagnostic view with truncation and metadata
impl<T: fmt::Display, const R: usize, const C: usize> fmt::Debug for Matrix<T, R, C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // configuration: cap rows/cols in debug
        const MAX_R: usize = 6;
        const MAX_C: usize = 8;

        let show_r = R.min(MAX_R);
        let show_c = C.min(MAX_C);
        let rows_elided = show_r < R;
        let cols_elided = show_c < C;

        writeln!(
            f,
            "Matrix<_, {R}, {C}> {{ rows: {R}, cols: {C}{}{} }}",
            if rows_elided { ", rows_truncated" } else { "" },
            if cols_elided { ", cols_truncated" } else { "" }
        )?;

        if R == 0 || C == 0 {
            writeln!(f, "[]")?;
            return Ok(());
        }

        // global max width computed from the shown window only (cheap for debug)
        let mut maxw = 0usize;
        for r in 0..show_r {
            for c in 0..show_c {
                let s = format!("{}", self.data[r][c]);
                maxw = maxw.max(s.len());
            }
        }
        // also consider the ellipsis symbol width for alignment
        let ell = "…";
        maxw = maxw.max(ell.len());

        // emit shown rows
        for r in 0..show_r {
            write!(f, "[")?;
            for c in 0..show_c {
                if c > 0 {
                    write!(f, " ")?;
                }
                let s = format!("{}", self.data[r][c]);
                write!(f, "{:>width$}", s, width = maxw)?;
            }
            if cols_elided {
                write!(f, " {} (+{} cols)", ell, C - show_c)?;
            }
            writeln!(f, "]")?;
        }

        // indicate omitted rows
        if rows_elided {
            write!(f, "[")?;
            for c in 0..show_c {
                if c > 0 {
                    write!(f, " ")?;
                }
                write!(f, "{:>width$}", ell, width = maxw)?;
            }
            if cols_elided {
                write!(f, " {} (+{} cols)", ell, C - show_c)?;
            }
            writeln!(f, "] (+{} rows)", R - show_r)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::Matrix;
    use rand::distr::{Distribution, StandardUniform};
    use rand::{Rng, SeedableRng, rngs::StdRng};

    fn random_matrix<T, const R: usize, const C: usize>(rng: &mut impl Rng) -> Matrix<T, R, C>
    where
        T: Default + Copy,
        StandardUniform: Distribution<T>,
    {
        #[allow(unused_mut)]
        let mut m = Matrix::<T, R, C>::new();
        for i in 0..R {
            for j in 0..C {
                let val: T = rng.random();
                unsafe {
                    let ptr = (&m as *const _ as *mut Matrix<T, R, C>).as_mut().unwrap();
                    (*ptr).data[i][j] = val;
                }
            }
        }
        m
    }

    #[test]
    fn random_fill_and_get_invariants() {
        let mut rng = StdRng::seed_from_u64(0xDEADBEEF);

        for _ in 0..200 {
            let r = rng.random_range(1..=8);
            let c = rng.random_range(1..=8);

            macro_rules! case {
                ($rr:expr, $cc:expr) => {{
                    if r == $rr && c == $cc {
                        let m = random_matrix::<i32, { $rr }, { $cc }>(&mut rng);
                        let mut checksum = 0i64;
                        for i in 0..$rr {
                            for j in 0..$cc {
                                let v = *m.get(i, j) as i64;
                                checksum += ((i as i64 + 31) * (j as i64 + 17)) ^ v;
                            }
                        }
                        assert!(checksum != 0 || ($rr * $cc == 0));
                        return;
                    }
                }};
            }

            case!(1, 1);
            case!(1, 2);
            case!(2, 1);
            case!(2, 2);
            case!(2, 3);
            case!(3, 2);
            case!(3, 3);
            case!(4, 4);
            case!(4, 7);
            case!(7, 4);
            case!(8, 8);

            {
                let m = random_matrix::<i32, 3, 5>(&mut rng);
                let mut checksum = 0i64;
                for i in 0..3 {
                    for j in 0..5 {
                        checksum ^= (*m.get(i, j) as i64) + (i as i64) * 13 + (j as i64) * 7;
                    }
                }
                assert!(checksum != 0);
            }
        }
    }

    #[test]
    fn display_uniform_width() {
        let mut rng = StdRng::seed_from_u64(12345);
        let m = random_matrix::<i64, 3, 4>(&mut rng);
        let s = format!("{}", m);

        assert!(s.contains("3x4"));
        let lines: Vec<_> = s.lines().collect();
        assert!(lines.len() >= 4);
        for line in &lines[1..] {
            assert!(line.starts_with('[') && line.ends_with(']'));
        }
        for line in &lines[1..] {
            let inner = &line[1..line.len() - 1];
            let tokens: Vec<&str> = inner.split(' ').filter(|t| !t.is_empty()).collect();
            assert_eq!(tokens.len(), 4);
        }
    }
    #[test]
    fn display_stdout() {
        let mut rng = StdRng::seed_from_u64(12345);
        let m = random_matrix::<i64, 3, 4>(&mut rng);
        println!("{m}");
    }
    #[test]
    fn debug_stdout() {
        let mut rng = StdRng::seed_from_u64(12345);
        let m = random_matrix::<u8, 3, 4>(&mut rng);
        println!("{m:?}");
    }
}
