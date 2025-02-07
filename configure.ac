AC_INIT([service-compute-cuda], [0.1])

# Force autoconf to be at least this version number.
AC_PREREQ([2.68])

# Put autotools auxiliary files in a subdir, so they don't clutter top dir.
# I am not actually doing this at the moment because there should be relatively
# little in the top directory anyway.
#AC_CONFIG_AUX_DIR([build-aux])
AC_CONFIG_MACRO_DIR([m4])

AM_SILENT_RULES([yes])

# Enable "automake" to simplify creating makefiles. foreign relaxes some GNU
# checks. -Wall and -Werror are instructions to Automake, not gcc.
AM_INIT_AUTOMAKE([foreign subdir-objects -Wall -Werror])

# These are the files to be generated.
AC_CONFIG_FILES([Makefile])
PKG_INSTALLDIR
AC_CONFIG_FILES([fractos-service-compute-cuda.pc])
AC_CONFIG_FILES([src/server/Makefile])
AC_CONFIG_FILES([src/library/Makefile])
AC_CONFIG_FILES([src/include/Makefile])

# Put configuration results in this file for us to #include in our program.
AC_CONFIG_HEADERS([config.h])

# Check for a C++ compiler.
if test -z "$CXXFLAGS"; then
    reset_cxxflags=1
fi
AC_PROG_CXX
AM_PROG_AR
LT_INIT()
AC_LANG([C++])
if test "$reset_cxxflags" = "1"; then
    CXXFLAGS=
fi
AX_CHECK_COMPILE_FLAG([-std=c++20],     [CXXFLAGS="$CXXFLAGS -std=c++20"], [exit 1])
AX_CHECK_COMPILE_FLAG([-Wall],          [CXXFLAGS="$CXXFLAGS -Wall"])
AX_CHECK_COMPILE_FLAG([-g],             [CXXFLAGS="$CXXFLAGS -g"])
AX_CHECK_COMPILE_FLAG([-rdynamic],      [CXXFLAGS="$CXXFLAGS -rdynamic"])
AX_CHECK_COMPILE_FLAG([-fvisibility=hidden], [CXXFLAGS="$CXXFLAGS -fvisibility=hidden"])

CONFIG_STR=""

# Configurable options

AC_ARG_ENABLE(debug, AS_HELP_STRING([--enable-debug],
              [enable various debugging aids, default: no]))

PKG_CONFIG_CFLAGS=
if test "x$enable_debug" = "xyes" ; then
    AX_CHECK_COMPILE_FLAG([-O0],     [CXXFLAGS="$CXXFLAGS -O0"], [exit 1])
    CONFIG_STR="$CONFIG_STR-debug"
else
    AX_CHECK_COMPILE_FLAG([-fcf-protection=none],[CXXFLAGS="$CXXFLAGS -fcf-protection=none"])
    AX_CHECK_COMPILE_FLAG([-fno-stack-protector],[CXXFLAGS="$CXXFLAGS -fno-stack-protector"])
    AX_CHECK_COMPILE_FLAG([-Ofast],[CXXFLAGS="$CXXFLAGS -Ofast"], [exit 1])
    PKG_CONFIG_CFLAGS=-DFRACTOS_NDEBUG
    AX_CHECK_COMPILE_FLAG([-DNDEBUG],[CXXFLAGS="$CXXFLAGS -DNDEBUG $PKG_CONFIG_CFLAGS"], [exit 1])
fi
AC_SUBST(PKG_CONFIG_CFLAGS)

# Check for libraries, headers, data etc here.

PKG_CPPFLAGS=
PKG_LDFLAGS=
PKG_LIBS=

PKG_CHECK_MODULES([FRACTOS_CORE], [fractos-core])
PKG_CPPFLAGS="$PKG_CPPFLAGS $FRACTOS_CORE_CFLAGS"
PKG_LIBS="$PKG_LIBS $FRACTOS_CORE_LIBS"

PKG_CHECK_MODULES([FRACTOS_COMMON], [fractos-common])
PKG_CPPFLAGS="$PKG_CPPFLAGS $FRACTOS_COMMON_CFLAGS"
PKG_LIBS="$PKG_LIBS $FRACTOS_COMMON_LIBS"

PKG_CHECK_MODULES([GLOG], [libglog])
PKG_CPPFLAGS="$PKG_CPPFLAGS $GLOG_CFLAGS"
PKG_LIBS="$PKG_LIBS $GLOG_LIBS"

AX_BOOST_BASE([1.65], [], [exit 1])
PKG_CPPFLAGS="$PKG_CPPFLAGS $BOOST_CPPFLAGS"
PKG_LDFLAGS="$PKG_LDFLAGS $BOOST_LDFLAGS"

AX_BOOST_PROGRAM_OPTIONS()
if test "$BOOST_PROGRAM_OPTIONS_LIB" = ""; then
   exit 1
fi
PKG_LIBS="$PKG_LIBS $BOOST_PROGRAM_OPTIONS_LIB"

# NOTE: set at the very end so 3rd-party tests don't fail with it
AX_CHECK_COMPILE_FLAG([-Werror],        [CXXFLAGS="$CXXFLAGS -Werror"])

AC_SUBST(PKG_CPPFLAGS)
AC_SUBST(PKG_LDFLAGS)
AC_SUBST(PKG_LIBS)

AC_SUBST(CONFIG_STR)

# Last line: actually generate all required output files.
AC_OUTPUT
