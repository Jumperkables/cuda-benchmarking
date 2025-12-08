#pragma once

template <typename T>
constexpr T ceil_div(T num, T den) {
    return (num + den - 1) / den;
}