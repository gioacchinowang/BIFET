// timer module
// timer record time elapse from Timer::tick to Timer::tock
// timer support nested/overlaped recording

#ifndef BIFET_TIMR_H
#define BIFET_TIMR_H

#include <cassert>
#include <chrono>
#include <ctime>
#include <iostream>
#include <map>
#include <string>

class Timer {
  typedef std::chrono::time_point<std::chrono::high_resolution_clock> tp;
  typedef std::chrono::duration<double, std::milli> duration;
  typedef std::map<std::string, std::pair<tp, double>> timecache;
#ifndef NDEBUG
public:
#endif
  timecache record;

public:
  Timer() = default;
  Timer(const Timer &) = delete;
  Timer &operator=(const Timer &) = delete;
  Timer &operator=(Timer &&) = delete;
  virtual ~Timer() = default;
  // record start point of timer
  // 1st argument: name of timing record
  inline void tick(std::string flag) {
    record[flag].first = std::chrono::high_resolution_clock::now();
  }
  // record end point of timer
  // then calculate corresponding elapsed time
  // 1st argument: name of timing record
  inline void tock(std::string flag) {
    duration diff =
        (std::chrono::high_resolution_clock::now() - record[flag].first);
    record[flag].second = diff.count();
  }
  // print to stdout the elapsed time in ms resolution
  // 1st argument: (optional) name of timing record
  // if no name provided, print all records
  inline void print(std::string flag = std::string()) {
    if (flag.empty()) {
      for (auto i : record)
        std::cout << i.first << " " << i.second.second << " ms" << std::endl;
    } else {
      auto target = record.find(flag);
      assert(target != record.end());
      std::cout << target->first << " " << target->second.second << " ms"
                << std::endl;
    }
  }
};

#endif
