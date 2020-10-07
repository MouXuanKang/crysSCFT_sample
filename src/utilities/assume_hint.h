#pragma once
#ifdef __GNUC__
#ifndef __assume
#define __assume(cond) do { if (!(cond)) __builtin_unreachable(); } while (0)
#endif // __assume
#endif
