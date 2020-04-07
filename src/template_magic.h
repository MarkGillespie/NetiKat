// Function to take template arguments and construct a vector containing them.
// See
// https://stackoverflow.com/questions/15581999/variadic-template-function-specialize-head-tail-and-empty-base-case
// Using typename instead of size_t in the first function magically makes it
// work

// Class to take template arguments and construct a vector containing them.
// See
// https://stackoverflow.com/questions/15581999/variadic-template-function-specialize-head-tail-and-empty-base-case
// Using typename instead of size_t in the first function magically makes it
// work
// TODO: append to back of list instead of front
struct TemplateToVec {
  template <typename... ns> static std::vector<size_t> parse();
  template <size_t n, size_t... ns> static std::vector<size_t> parse();
};

template <typename... ns> std::vector<size_t> TemplateToVec::parse() {
  return std::vector<size_t>();
}
template <size_t n, size_t... ns> std::vector<size_t> TemplateToVec::parse() {
  std::vector<size_t> vec = parse<ns...>();
  vec.insert(std::begin(vec), n);
  return vec;
}

// A bunch of structs that work together to multiply together integers that are
// given as template arguments
// call like TemplateProduct<2, 3, 4>::value
template <size_t... ns> struct TypeTuple {};

template <size_t, class> struct TemplateBoundedProduct;

template <size_t... ns> struct TemplateBoundedProduct<0, TypeTuple<ns...>> {
  static constexpr size_t value = 1;
};

template <size_t k, size_t n, size_t... ns>
struct TemplateBoundedProduct<k, TypeTuple<n, ns...>> {
  static constexpr size_t value =
      n * TemplateBoundedProduct<k - 1, TypeTuple<ns...>>::value;
};

template <size_t... ns> struct TemplateProduct {
  static constexpr size_t value =
      TemplateBoundedProduct<sizeof...(ns), TypeTuple<ns...>>::value;
};
