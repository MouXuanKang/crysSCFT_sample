#pragma once
template<class T, size_t N>
struct c_array
{
	T arr[N];

	constexpr T const& operator[](size_t p) const
	{
		return arr[p];
	}

	constexpr T const* begin() const
	{
		return arr + 0;
	}
	constexpr T const* end() const
	{
		return arr + N;
	}

	constexpr size_t size() const
	{
		return N;
	}
};

template<class T>
struct c_array<T, 0> {};

