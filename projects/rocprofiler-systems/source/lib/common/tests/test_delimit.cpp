// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/delimit.hpp"

#include <gtest/gtest.h>
#include <string>
#include <vector>

using namespace rocprofsys::common;

using strvec = std::vector<std::string>;

TEST(delimit_test, basic_split)
{
    EXPECT_EQ(delimit("a,b,c", ","), (strvec{ "a", "b", "c" }));
}

TEST(delimit_test, default_delimiters)
{
    // default set is "\"',;: " (space, comma, semicolon, colon, quotes)
    EXPECT_EQ(delimit("a b,c"), (strvec{ "a", "b", "c" }));
}

TEST(delimit_test, empty_tokens_dropped)
{
    EXPECT_EQ(delimit("a,,b", ","), (strvec{ "a", "b" }));
}

TEST(delimit_test, leading_trailing_delimiters)
{
    EXPECT_EQ(delimit(",a,b,", ","), (strvec{ "a", "b" }));
}

TEST(delimit_test, multi_char_delimiter_set)
{
    // every character in the delimiter set is an independent separator
    EXPECT_EQ(delimit("a,b;c d", ",; "), (strvec{ "a", "b", "c", "d" }));
}

TEST(delimit_test, no_delimiter_present)
{
    EXPECT_EQ(delimit("abc", ","), (strvec{ "abc" }));
}

TEST(delimit_test, empty_input) { EXPECT_EQ(delimit("", ","), strvec{}); }

TEST(delimit_test, predicate_prefixes_each_token)
{
    auto _pred = [](const std::string& _v) { return "component::" + _v; };
    EXPECT_EQ(delimit("x y", " ", _pred), (strvec{ "component::x", "component::y" }));
}

TEST(delimit_test, predicate_applied_after_empty_drop)
{
    // empty tokens are dropped before the predicate runs
    auto _pred = [](const std::string& _v) { return "component::" + _v; };
    EXPECT_EQ(delimit(",x,", ",", _pred), (strvec{ "component::x" }));
}
