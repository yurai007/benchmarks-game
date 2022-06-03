// The Computer Language Benchmarks Game
// https://salsa.debian.org/benchmarksgame-team/benchmarksgame/
//
// Mixed implementation where main idea comes from [1] with core functionality replaced with [2].
// On my computer Intel Core i7-6500U CPU @ 2.50GHz it's ~10% faster than current top solution [3]
// and only ~50% slower than pure copying (cp) files.
//
// [1] https://benchmarksgame-team.pages.debian.net/benchmarksgame/program/revcomp-gpp-7.html
//     contributed by roman blog
// [2] https://benchmarksgame-team.pages.debian.net/benchmarksgame/program/revcomp-gpp-2.html
//     contributed by Adam Kewley
// [3] https://benchmarksgame-team.pages.debian.net/benchmarksgame/program/revcomp-gcc-7.html
//     contributed by Jeremy Zerfas

#include <limits>
#include <array>
#include <unistd.h>
#undef NDEBUG
#include <cassert>
#include <string_view>
#include <vector>
#include <fcntl.h>
#include <compare>
#include <sys/sendfile.h>
#include <immintrin.h>
#include <x86intrin.h>
#include <cstring>
#include <cstdint>

using sv = std::string_view;
using namespace std::literals;

constexpr uint8_t swmap(uint8_t c) {
  switch(c) {
  case 'A': case 'a': return 'T';// 'A' | 'a' => 'T',
  case 'C': case 'c': return 'G';// 'C' | 'c' => 'G',
  case 'G': case 'g': return 'C';// 'G' | 'g' => 'C',
  case 'T': case 't': return 'A';// 'T' | 't' => 'A',
  case 'U': case 'u': return 'A';// 'U' | 'u' => 'A',
  case 'M': case 'm': return 'K';// 'M' | 'm' => 'K',
  case 'R': case 'r': return 'Y';// 'R' | 'r' => 'Y',
  case 'W': case 'w': return 'W';// 'W' | 'w' => 'W',
  case 'S': case 's': return 'S';// 'S' | 's' => 'S',
  case 'Y': case 'y': return 'R';// 'Y' | 'y' => 'R',
  case 'K': case 'k': return 'M';// 'K' | 'k' => 'M',
  case 'V': case 'v': return 'B';// 'V' | 'v' => 'B',
  case 'H': case 'h': return 'D';// 'H' | 'h' => 'D',
  case 'D': case 'd': return 'H';// 'D' | 'd' => 'H',
  case 'B': case 'b': return 'V';// 'B' | 'b' => 'V',
  case 'N': case 'n': return 'N';// 'N' | 'n' => 'N',
  default: return '_';
  }
}

constexpr auto map = ([] {
  constexpr auto max = std::numeric_limits<uint8_t>::max() + size_t{1};
  std::array<uint16_t, max * max> map{};
  for(size_t it = 0; it < map.size(); ++it) {
    uint8_t hi = (it >> 8), lo = it;
    map[it] = (swmap(lo) << 8) | (swmap(hi));
  }
  return map;
})();

constexpr auto map256 = ([] {
  constexpr auto max = std::numeric_limits<uint8_t>::max() + size_t{1};
  std::array<uint8_t, max> map{};
  for(size_t it = 0; it < max; ++it)
    map[it] = swmap(it);
  return map;
})();

struct range{
  size_t begin{}, size{};
  auto operator<=>(const range &) const = default;
};

constexpr auto simd_width = 16;

static void reverse_complement_sse(const char *in, char *out) {
  using reg = __m128i;
  auto packed = [](char c) {
    return _mm_set_epi8(c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c);
  };
  // reverse elements in the registers, both in and out can be unaligned
  reg input = _mm_loadu_si128(reinterpret_cast<const reg*>(in));
  reg v =  _mm_shuffle_epi8(input, _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15));

  // AND all elements with 0x1f (11111)b, so that a smaller LUT (< 32 bytes)
  // can be used. This is important with SIMD because, unlike
  // single-char complement (above), SIMD uses 16-byte shuffles. The
  // single-char LUT would require four shuffles, this LUT requires
  // two.
  v = _mm_and_si128(v, packed(0x1f));

  // Lookup for all v elements < 16
  reg lt16_mask = _mm_cmplt_epi8(v, packed(16));
  reg lt16_els = _mm_and_si128(v, lt16_mask);
  reg lt16_lut = _mm_set_epi8('\0', 'N', 'K', '\0', 'M', '\n', '\0', 'D',
                              'C', '\0', '\0', 'H', 'G', 'V', 'T', '\0');
  reg lt16_vals = _mm_shuffle_epi8(lt16_lut, lt16_els);

  // Lookup for all elements >16
  reg g16_els = _mm_sub_epi8(v, packed(16));
  reg g16_lut = _mm_set_epi8('\0', '\0', '\0', '\0', '\0', '\0', 'R', '\0',
                             'W', 'B', 'A', 'A', 'S', 'Y', '\0', '\0');
  reg g16_vals = _mm_shuffle_epi8(g16_lut, g16_els);

  // OR both lookup results - merge both vectors
  reg res = _mm_or_si128(lt16_vals, g16_vals);
  _mm_storeu_si128(reinterpret_cast<reg*>(out), res);
}

static void replace60_simd(const char *in, char *out, int offset) {
  auto size = ((60 - offset) / 2);
  auto fast_op = [&in, &out](auto working_set){
    const int n = working_set / (simd_width / 2);
    for (auto i = 0; i < n; i++) {
      in -= simd_width;
      reverse_complement_sse(in, out);
      out += simd_width;
    }
    uint16_t from;
    for (auto i = 0; i < working_set - (simd_width / 2) * n; i++) {
      in -= 2;
      // don't bother UBSan
      std::memcpy(&from, in, sizeof(uint16_t));
      auto tmp = map[from];
      std::memcpy(out, &tmp, sizeof(uint16_t));
      out += 2;
    }
  };

  fast_op(size);

  if (offset % 2) {
    //   ...1\n
    //   0...
    *out++ = map256[*(--in)];
    --in;
    //     assert(*in == '\n');
    *out++ = map256[*(--in)];

    fast_op(29 - size);
  } else {// even
    //   ...\n
    //   ...
    in -= 1;
    //     assert(*in == '\n');
    fast_op(30 - size);
  }
  *(out++) = '\n';
}

static void replace(int fd, range r) {
  auto off = (60 - (r.size % 61));
  constexpr size_t line_size = 61;
  constexpr size_t block_size = line_size * 1024;
  char buf[block_size]{};
  char outbuf[block_size]{};
  auto nblock = r.size / block_size;
  auto tail = r.size - (nblock * block_size);

  // handle block after block
  for(size_t n = 1; n <= nblock; ++n) {
    pread(fd, buf, block_size, r.begin + r.size - n * block_size);
    auto it = std::end(buf), oit = std::begin(outbuf), oend = std::end(outbuf);
    while(oit < oend) {
      replace60_simd(it, oit, off);
      it -= line_size;
      oit += line_size;
    }
    write(STDOUT_FILENO, outbuf, block_size);
  }

  pread(fd, buf, tail, r.begin);
  auto it = std::begin(buf) + tail, oit = std::begin(outbuf);

  // handle remaining block
  for(size_t n = 0; n < tail / line_size; ++n) {
    replace60_simd(it, oit, off);
    it -= line_size;
    oit += line_size;
  }

  // handle remaining line
  for(size_t n = 0; n < (tail - (tail / line_size) * line_size); ++n) {
    *oit++ = swmap(*(--it));
  }

  write(STDOUT_FILENO, outbuf, tail);
  write(STDOUT_FILENO, "\n", 1);
}

static auto find_first_of(int fd, char c, size_t pos, size_t & endfile) {
  constexpr auto block_size = 1024u * 32u;
  uint8_t mem[block_size]{};
  if (pos == sv::npos)
    return pos;
  while (true) {
    auto bytes = pread(fd, mem, block_size, pos);
    assert(bytes >= 0);
    if (!bytes) {
      endfile = pos;
      return sv::npos;
    }
    auto r = sv{(const char *)mem, size_t(bytes)}.find_first_of(c);
    if (r != sv::npos)
      return pos + r;
    pos += bytes;
  }
}

int main() {
  int fd = open("/dev/stdin", O_RDONLY);
  assert(fd != -1);

  auto next_group = [=, prev = 0ul]() mutable -> std::pair<range, range> {
    size_t endfile = 0;
    auto arrow_pos = find_first_of(fd, '>', prev, endfile);
    auto begin_pos = find_first_of(fd, '\n', arrow_pos, endfile);
    if (begin_pos == sv::npos)
      return {};
    prev = find_first_of(fd, '>', begin_pos, endfile);
    prev = (prev == sv::npos) ? endfile : prev;
    return {{arrow_pos, begin_pos - arrow_pos + 1}, {begin_pos + 1, prev - begin_pos - 2}};
  };

  std::vector<std::pair<range, range>> groups;
  for (auto pair = next_group(); pair != std::pair<range, range>{}; pair = next_group())
    groups.emplace_back(pair);

  for(auto [header, group]: groups) {
    off_t begin = header.begin;
    sendfile(STDOUT_FILENO, fd, &begin, header.size);
    replace(fd, group);
  };
  return 0;
}
