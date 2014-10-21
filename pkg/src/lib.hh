#ifndef LIB_HH
#define LIB_HH

#include "env.hh"
#include "matrix.hh"

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_psi.h>
#include <gsl/gsl_sf.h>

class PopLib {
public:
  static void set_dir_exp(const Matrix &u, Matrix &exp);
  static void set_dir_exp(const D3 &u, D3 &exp);
  static void set_dir_exp(uint32_t a, const Matrix &u, Matrix &exp);
};

inline void
PopLib::set_dir_exp(const Matrix &u, Matrix &exp)
{
  const double ** const d = u.data();
  double **e = exp.data();
  for (uint32_t i = 0; i < u.m(); ++i) {
    double s = .0;
    for (uint32_t j = 0; j < u.n(); ++j) 
      s += d[i][j];
    assert (s > .0);
    double psi_sum = gsl_sf_psi(s);
    for (uint32_t j = 0; j < u.n(); ++j) {
      double v = d[i][j];
      e[i][j] = gsl_sf_psi(v) - psi_sum;
    }
  }
}

inline void
PopLib::set_dir_exp(const D3 &u, D3 &exp)
{
  const double *** const d = u.data();
  double ***e = exp.data();

  for (uint32_t i = 0; i < u.m(); ++i) {
    for (uint32_t j = 0; j < u.n(); ++j) {
      double s = .0;
      for (uint32_t k = 0; k < u.k(); ++k) 
	s += d[i][j][k];
      assert (s > .0);
      double psi_sum = gsl_sf_psi(s);
      for (uint32_t k = 0; k < u.k(); ++k) {
	assert (d[i][j][k] > .0);
	double v = d[i][j][k];
	e[i][j][k] = gsl_sf_psi(v) - psi_sum;
      }
    }
  }
}

inline void
PopLib::set_dir_exp(uint32_t a, const Matrix &u, Matrix &exp)
{
  const double ** const d = u.data();
  double **e = exp.data();

  double s = .0;
  for (uint32_t j = 0; j < u.n(); ++j) 
    s += d[a][j];
  double psi_sum = gsl_sf_psi(s);
  for (uint32_t j = 0; j < u.n(); ++j) 
    e[a][j] = gsl_sf_psi(d[a][j]) - psi_sum;
}

#endif
