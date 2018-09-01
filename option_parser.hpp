#include <string>
#include <sstream>
#include <optional>
#include <vector>
#include <unordered_map>
#include <cstddef>
#include <functional>

using Int = ptrdiff_t;
using Uint = size_t;

template<typename Map>
typename Map::mapped_type* map_element_ptr(
  Map& map, const typename Map::key_type& key)
{
  auto it = map.find(key);
  return it == map.end() ? NULL : &it->second;
}

//TODO add --help flag handling
class OptionParser
{
public:
  struct Result
  {
    bool data_valid;
    bool error;
    std::string message;
    operator bool() const { return data_valid; }
  };

  struct Switch
  {
    std::string description;
    bool* dst;
  };

  struct Option
  {
    std::string name;
    std::string description;
    std::function<bool(const char*)> handler;
    Int min_num = 0;
    Int max_num = 1;
    Int num = 0;
  };

  void add_switch(
    const std::string_view& name, const std::string_view& description,
    bool* dst)
  {
    switches.emplace(std::string(name), Switch{std::string(description), dst});
  }

  template<typename T>
  void add_optional_flag(
    const std::string_view& name, const std::string_view& description,
    std::optional<T>* dst)
  {
    Option option;
    option.name = std::string(name);
    option.description = std::string(description);
    option.handler = [dst](const char* val)
    {
      std::stringstream s(val);
      T converted;
      s >> converted;
      if(s.fail()) return false;
      *dst = converted;
      return true;
    };

    flags.emplace(std::string(name), std::move(option));
  }

  template<typename T>
  void add_optional_flag(
    const std::string_view& name, const std::string_view& description,
    T* dst)
  {
    Option option;
    option.name = std::string(name);
    option.description = std::string(description);
    option.handler = [dst](const char* val)
    {
      std::stringstream s(val);
      T converted;
      s >> converted;
      if(s.fail()) return false;
      *dst = converted;
      return true;
    };

    flags.emplace(std::string(name), std::move(option));
  }

  template<typename T>
  void add_positional(
    const std::string_view& name, const std::string_view& description,
    T* dst)
  {
    Option option;
    option.name = std::string(name);
    option.description = std::string(description);
    option.min_num = 1;
    option.handler = [dst](const char* val)
    {
      std::stringstream s(val);
      T converted;
      s >> converted;
      if(s.fail()) return false;
      *dst = converted;
      return true;
    };

    positional.push_back(std::move(option));
  }

  template<typename T>
  void add_multi_positional(
    const std::string_view& name, const std::string_view& description,
    Int min_num, Int max_num,
    std::vector<T>* dst)
  {
    Option option;
    option.name = std::string(name);
    option.description = std::string(description);
    option.min_num = min_num;
    option.max_num = max_num;
    option.handler = [dst](const char* val)
    {
      std::stringstream s(val);
      T converted;
      s >> converted;
      if(s.fail()) return false;
      dst->push_back(converted);
      return true;
    };

    positional.push_back(std::move(option));
  }

  Result parse(int argc, char** argv)
  {
    for(auto& [name, switch_] : switches) *switch_.dst = false;

    auto positional_it = positional.begin();

    for(int i = 1; i < argc; i++)
    {
      if(argv[i][0] == '-')
      {
        if(auto switch_ = map_element_ptr(switches, argv[i]))
          *switch_->dst = true;
        else if(auto flag = map_element_ptr(flags, argv[i]))
        {
          std::string name = argv[i];
          if(++i == argc) return fail(name, " requires an argument");

          if(++flag->num > flag->max_num)
            return fail(
              "Too many ", name, " flags. There can be at most ",
              flag->max_num);

          if(!flag->handler(argv[i]))
            return fail(argv[i], " is not a valid value for ", name);
        }
        else
          return fail("Flag ", argv[i], " is not supported.");
      }
      else if(positional_it < positional.end())
      {
        if(!positional_it->handler(argv[i]))
          return fail(
            argv[i], " is not a valid value "
            "for the positional argument ", positional_it->name);

        if(++positional_it->num == positional_it->max_num) positional_it++;
      }
      else
        return fail("Too many positional arguments.");
    }

    for(auto& [name, option] : flags)
      if(option.num < option.min_num)
        return fail(
          "There should be at least ", option.min_num, " flags ",
          name, ", but there are only ", option.num, ".");

    for(auto& option : positional)
      if(option.num < option.min_num)
        return fail(
          "There should be at least ", option.min_num,
          " positional arguments ", option.name, ", but there are only ",
          option.num, ".");

    return Result{true, false, ""};
  }


private:  
  
  template<typename... Args>
  Result fail(const Args&... args)
  {
    Result r;
    r.data_valid = false;
    r.error = true;
    std::stringstream s;
    s << "Failed to parse command line arguments: ";
    (s << ... << args);
    r.message = s.str();
    return r;
  }

  std::unordered_map<std::string, Switch> switches;
  std::unordered_map<std::string, Option> flags;
  std::vector<Option> positional;
};

