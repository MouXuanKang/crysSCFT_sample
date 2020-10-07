#pragma once
#define OMP_SCHEDULE schedule(static)
#define OMP_SIMDLEN simdlen(8)
#define OMP_SIMDLEN_SCHEDULE OMP_SIMDLEN OMP_SCHEDULE
