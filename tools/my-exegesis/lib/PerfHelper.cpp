#include "PerfHelper.h"
#ifdef HAVE_LIBPFM
#include "perfmon/perf_event.h"
#include "perfmon/pfmlib.h"
#include "perfmon/pfmlib_perf_event.h"
#endif

namespace myexegesis {
namespace pfm {
#ifdef HAVE_LIBPFM
static bool isPfmError(int Code) { return Code != PFM_SUCCESS; }
#endif

bool pfmInitialize() {
#ifdef HAVE_LIBPFM
    return isPfmError(pfm_initialize());
#else
    return true;
#endif
}

void pfmTerminate() {
#ifdef HAVE_LIBPFM
  pfm_terminate();
#endif
}
} // namespace pfm
} // namespace myexegesis