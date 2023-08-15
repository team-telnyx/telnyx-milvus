AC_DEFUN([FA_CHECK_HIP], [

AC_ARG_WITH(hip,
  [AS_HELP_STRING([--with-hip=<prefix>], [prefix of the HIP installation])])
AC_ARG_WITH(hip-arch,
  [AS_HELP_STRING([--with-hip-arch=<gencodes>], [device specific -gencode flags])],
  [],
  [with_hip_arch=default])

if test x$with_hip != xno; then
  if test x$with_hip != x; then
    hip_prefix=$with_hip
    AC_CHECK_PROG(HIPCC, [hipcc], [$hip_prefix/bin/hipcc], [], [$hip_prefix/bin])
    HIPCC_CPPFLAGS="-I$hip_prefix/include"
    HIPCC_LDFLAGS="-L$hip_prefix/lib"
  else
    AC_CHECK_PROGS(HIPCC, [hipcc /opt/rocm-5.3.0/hip/bin/hipcc], [])
    if test "x$HIPCC" == "x/opt/rocm-5.3.0/hip/bin/hipcc"; then
      hip_prefix="/opt/rocm-5.3.0/hip"
      HIPCC_CPPFLAGS="-I$hip_prefix/include"
      HIPCC_LDFLAGS="-L$hip_prefix/lib"
    else
      hip_prefix=""
      HIPCC_CPPFLAGS=""
      HIPCC_LDFLAGS=""
    fi
  fi

  if test "x$HIPCC" == x; then
    AC_MSG_ERROR([Couldn't find hipcc])
  fi

if test "x$with_hip_arch" == xdefault; then
  with_hip_arch="-gencode=arch=gfx900,code=sm_60 \
  -gencode=arch=gfx906,code=sm_61 \
  -gencode=arch=gfx908,code=sm_70 \
  -gencode=arch=gfx1030,code=sm_75"
fi

  fa_save_CPPFLAGS="$CPPFLAGS"
  fa_save_LDFLAGS="$LDFLAGS"
  fa_save_LIBS="$LIBS"

  CPPFLAGS="$HIPCC_CPPFLAGS $CPPFLAGS"
  LDFLAGS="$HIPCC_LDFLAGS $LDFLAGS"

AC_CHECK_HEADER([hip/hip_runtime.h], [], AC_MSG_FAILURE([Couldn't find hip/hip_runtime.h]))
AC_CHECK_LIB([hipblas], [hipblasAlloc], [], AC_MSG_FAILURE([Couldn't find libhipblas]))
AC_CHECK_LIB([hip_runtime], [hipSetDevice], [], AC_MSG_FAILURE([Couldn't find libhip_runtime]))


  HIPCC_LIBS="$LIBS"
  HIPCC_CPPFLAGS="$CPPFLAGS"
  HIPCC_LDFLAGS="$LDFLAGS"
  CPPFLAGS="$fa_save_CPPFLAGS"
  LDFLAGS="$fa_save_LDFLAGS"
  LIBS="$fa_save_LIBS"
fi

AC_SUBST(HIPCC)
AC_SUBST(HIPCC_CPPFLAGS)
AC_SUBST(HIPCC_LDFLAGS)
AC_SUBST(HIPCC_LIBS)
AC_SUBST(HIP_PREFIX, $hip_prefix)
AC_SUBST(HIP_ARCH, $with_hip_arch)
])
