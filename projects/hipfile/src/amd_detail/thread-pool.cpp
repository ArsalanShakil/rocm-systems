/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "thread-pool.h"

#include <taskflow/taskflow.hpp>

#include <atomic>
#include <exception>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace hipFile {

namespace {

    std::size_t defaultThreadCount() noexcept
    {
        const auto thread_count = std::thread::hardware_concurrency();
        return thread_count == 0 ? 1 : static_cast<std::size_t>(thread_count);
    }

}

struct ThreadPool::ThreadPoolStorage {
    tf::Executor executor{defaultThreadCount()};
};

class ThreadPool::TaskGroup : public ITaskGroup {
public:
    explicit TaskGroup(std::shared_ptr<ThreadPoolStorage> _storage) : storage{std::move(_storage)}
    {
    }

    ~TaskGroup() noexcept override
    {
        try {
            cancel_impl();
            wait_impl();
        }
        catch (...) {
            // Explicit wait() preserves task failures. Destructors cannot report them.
        }
    }

    void run(std::function<void()> work) override
    {
        if (!work) {
            throw std::invalid_argument("Task group work item cannot be empty");
        }

        std::lock_guard<std::mutex> lock{tasks_mutex};
        if (cancelled.load(std::memory_order_acquire)) {
            return;
        }

        tasks.push_back(storage->executor
                            .async([this, task = std::move(work)]() mutable {
                                if (!cancelled.load(std::memory_order_acquire)) {
                                    task();
                                }
                            })
                            .share());
    }

    void cancel() override
    {
        cancel_impl();
    }

    void wait() override
    {
        wait_impl();
    }

private:
    void cancel_impl() noexcept
    {
        cancelled.store(true, std::memory_order_release);
    }

    void wait_impl()
    {
        std::vector<std::shared_future<void>> current_tasks;
        {
            std::lock_guard<std::mutex> lock{tasks_mutex};
            current_tasks.swap(tasks);
        }

        std::exception_ptr first_exception{};
        for (auto &task : current_tasks) {
            try {
                task.get();
            }
            catch (...) {
                if (!first_exception) {
                    first_exception = std::current_exception();
                }
            }
        }

        {
            std::lock_guard<std::mutex> lock{tasks_mutex};
            if (tasks.empty()) {
                cancelled.store(false, std::memory_order_release);
            }
        }

        if (first_exception) {
            std::rethrow_exception(first_exception);
        }
    }

    std::shared_ptr<ThreadPoolStorage>    storage;
    std::mutex                            tasks_mutex;
    std::vector<std::shared_future<void>> tasks;
    std::atomic<bool>                     cancelled{false};
};

ThreadPool::ThreadPool() : storage{std::make_shared<ThreadPoolStorage>()}
{
}

ThreadPool::~ThreadPool() noexcept = default;

std::unique_ptr<ITaskGroup>
ThreadPool::makeTaskGroup()
{
    return std::make_unique<TaskGroup>(storage);
}

std::size_t
ThreadPool::threadCount() const noexcept
{
    return storage->executor.num_workers();
}

}
