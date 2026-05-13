/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "batch.h"
#include "buffer.h"
#include "context.h"
#include "file.h"
#include "hipfile.h"
#include "state.h"
#include "thread-pool.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace hipFile {

namespace {

    using InternalStatus = BatchOperation::InternalStatus;

    const char *name(hipFileStatus_t status) noexcept
    {
        switch (status) {
            case hipFileWaiting:
                return "hipFileWaiting";
            case hipFilePending:
                return "hipFilePending";
            case hipFileComplete:
                return "hipFileComplete";
            case hipFileCanceled:
                return "hipFileCanceled";
            case hipFileInvalid:
                return "hipFileInvalid";
            case hipFileTimeout:
                return "hipFileTimeout";
            case hipFileFailed:
                return "hipFileFailed";
            default:
                return "unknown hipFileStatus_t";
        }
    }

    const char *name(InternalStatus status) noexcept
    {
        switch (status) {
            case InternalStatus::Waiting:
                return "hipFileWaiting";
            case InternalStatus::Pending:
                return "hipFilePending";
            case InternalStatus::Running:
                return "hipFileRunning";
            case InternalStatus::Complete:
                return "hipFileComplete";
            case InternalStatus::Canceled:
                return "hipFileCanceled";
            case InternalStatus::Invalid:
                return "hipFileInvalid";
            case InternalStatus::Timeout:
                return "hipFileTimeout";
            case InternalStatus::Failed:
                return "hipFileFailed";
            default:
                return "unknown InternalStatus";
        }
    }

    hipFileStatus_t to_public(InternalStatus status) noexcept
    {
        switch (status) {
            case InternalStatus::Waiting:
                return hipFileWaiting;
            case InternalStatus::Pending:
            case InternalStatus::Running:
                return hipFilePending;
            case InternalStatus::Complete:
                return hipFileComplete;
            case InternalStatus::Canceled:
                return hipFileCanceled;
            case InternalStatus::Invalid:
                return hipFileInvalid;
            case InternalStatus::Timeout:
                return hipFileTimeout;
            case InternalStatus::Failed:
                return hipFileFailed;
            default:
                return hipFileInvalid;
        }
    }

    bool is_allowed_transition(InternalStatus from, InternalStatus to) noexcept
    {
        switch (from) {
            case InternalStatus::Waiting:
                return to == InternalStatus::Pending || to == InternalStatus::Invalid;
            case InternalStatus::Pending:
                return to == InternalStatus::Canceled || to == InternalStatus::Running;
            case InternalStatus::Running:
                return to == InternalStatus::Complete || to == InternalStatus::Failed ||
                       to == InternalStatus::Timeout;
            case InternalStatus::Canceled:
                return to == InternalStatus::Canceled;
            case InternalStatus::Complete:
            case InternalStatus::Failed:
            case InternalStatus::Invalid:
            case InternalStatus::Timeout:
                return false;
            default:
                return false;
        }
    }

    bool is_zero_timeout(const struct timespec *timeout) noexcept
    {
        return timeout != nullptr && timeout->tv_sec == 0 && timeout->tv_nsec == 0;
    }

    void validate_timeout(const struct timespec *timeout)
    {
        if (timeout == nullptr) {
            return;
        }
        if (timeout->tv_sec < 0 || timeout->tv_nsec < 0 || timeout->tv_nsec >= 1000000000L) {
            throw std::invalid_argument("Invalid batch status timeout");
        }
    }

    std::chrono::steady_clock::time_point timeout_deadline(const struct timespec *timeout)
    {
        return std::chrono::steady_clock::now() + std::chrono::seconds{timeout->tv_sec} +
               std::chrono::nanoseconds{timeout->tv_nsec};
    }

}

InvalidStateTransition::InvalidStateTransition(hipFileStatus_t from, hipFileStatus_t to)
    : InvalidStateTransition{name(from), name(to)}
{
}

InvalidStateTransition::InvalidStateTransition(const char *from, const char *to)
    : std::logic_error{std::string{"Invalid batch operation state transition: "} + from + " -> " + to}
{
}

BatchOperation::BatchOperation(std::unique_ptr<const hipFileIOParams_t> params,
                               std::shared_ptr<IBuffer> _buffer, std::shared_ptr<IFile> _file)
    : io_params{std::move(params)}, buffer{_buffer}, file{_file}
{
    // Cookie allows the user to track which operation caused the error.
    // It would be ideal if this could be passed as a member within the exception.

    // Check Buffer parameters
    if (io_params->u.batch.devPtr_base != buffer->getBuffer()) {
        throw std::invalid_argument("Buffer does not match buffer specified in io_params.");
    }
    if (io_params->u.batch.devPtr_offset < 0) {
        std::stringstream msg;
        msg << "Negative buffer offset specified. Value: " << io_params->u.batch.devPtr_offset;
        msg << ". Cookie: " << io_params->cookie;
        throw std::invalid_argument(msg.str());
    }
    if (buffer->getLength() <= static_cast<size_t>(io_params->u.batch.devPtr_offset)) {
        std::stringstream msg;
        msg << "Buffer offset exceeds the size of the buffer. Size: " << buffer->getLength();
        msg << ". Value: " << io_params->u.batch.devPtr_offset << ". Cookie: " << io_params->cookie;
        throw std::invalid_argument(msg.str());
    }
    if (buffer->getLength() - static_cast<size_t>(io_params->u.batch.devPtr_offset) <
        io_params->u.batch.size) {
        std::stringstream msg;
        msg << "IO Size exceeds the size of the buffer & offset. Buffer size: " << buffer->getLength();
        msg << ". Buffer offset: " << io_params->u.batch.devPtr_offset
            << ". IO size: " << io_params->u.batch.size;
        msg << ". Cookie: " << io_params->cookie;
        throw std::invalid_argument(msg.str());
    }

    // Check File parameters
    if (io_params->fh != file->handle()) {
        throw std::invalid_argument("File does not match handle specified in io_params.");
    }
    if (io_params->u.batch.file_offset < 0) {
        std::stringstream msg;
        msg << "Negative file offset specified. Value: " << io_params->u.batch.file_offset;
        msg << ". Cookie: " << io_params->cookie;
        throw std::invalid_argument(msg.str());
    }

    // Check OpCode
    if (io_params->opcode != hipFileBatchRead && io_params->opcode != hipFileBatchWrite) {
        std::stringstream msg;
        msg << "Bad opcode specified. Value: " << io_params->opcode;
        msg << ". Cookie: " << io_params->cookie;
        throw std::invalid_argument(msg.str());
    }

    // Check Batch Mode
    if (io_params->mode != hipFileBatch) {
        std::stringstream msg;
        msg << "Invalid Batch mode specified. Value: " << io_params->mode;
        msg << ". Cookie: " << io_params->cookie;
        throw std::invalid_argument(msg.str());
    }
}

void
BatchOperation::transition_to(InternalStatus next, ssize_t next_ret)
{
    if (!is_allowed_transition(status, next)) {
        throw InvalidStateTransition{name(status), name(next)};
    }

    status = next;
    ret    = next_ret;
}

void
BatchOperation::transition_to(InternalStatus next)
{
    if (!is_allowed_transition(status, next)) {
        throw InvalidStateTransition{name(status), name(next)};
    }

    status = next;
}

void
BatchOperation::mark_pending()
{
    std::lock_guard<std::mutex> lock{state_mutex};

    transition_to(InternalStatus::Pending);
}

void
BatchOperation::try_cancel()
{
    std::lock_guard<std::mutex> lock{state_mutex};

    try {
        transition_to(InternalStatus::Canceled);
    }
    catch (...) {
    }
}

hipFileIOEvents_t
BatchOperation::event() const
{
    std::lock_guard<std::mutex> lock{state_mutex};
    return {io_params->cookie, to_public(status), static_cast<size_t>(ret)};
}

bool
BatchOperation::is_terminal() const
{
    std::lock_guard<std::mutex> lock{state_mutex};
    return is_terminal_status(status);
}

bool
BatchOperation::is_terminal_status(InternalStatus s) noexcept
{
    switch (s) {
        case InternalStatus::Complete:
        case InternalStatus::Canceled:
        case InternalStatus::Invalid:
        case InternalStatus::Timeout:
        case InternalStatus::Failed:
            return true;
        case InternalStatus::Waiting:
        case InternalStatus::Pending:
        case InternalStatus::Running:
            return false;
        default:
            return false;
    }
}

void
BatchOperation::run()
{
    {
        std::lock_guard<std::mutex> lock{state_mutex};
        if (status == InternalStatus::Canceled) {
            return;
        }
        transition_to(InternalStatus::Running);
    }

    ssize_t result = 0;
    if (io_params->opcode == hipFileBatchRead) {
        result = hipFileRead(io_params->fh, io_params->u.batch.devPtr_base, io_params->u.batch.size,
                             io_params->u.batch.file_offset, io_params->u.batch.devPtr_offset);
    }
    else {
        result = hipFileWrite(io_params->fh, io_params->u.batch.devPtr_base, io_params->u.batch.size,
                              io_params->u.batch.file_offset, io_params->u.batch.devPtr_offset);
    }

    std::lock_guard<std::mutex> lock{state_mutex};
    transition_to(result >= 0 ? InternalStatus::Complete : InternalStatus::Failed, result);
}

void
BatchOperation::record_internal_error()
{
    std::lock_guard<std::mutex> lock{state_mutex};

    if (status == InternalStatus::Canceled) {
        return;
    }

    status = InternalStatus::Failed;
    ret    = -hipFileInternalError;
}

std::shared_ptr<IBatchOperation>
BatchOperationFactory::create(std::unique_ptr<const hipFileIOParams_t> params,
                              std::shared_ptr<IBuffer> buffer, std::shared_ptr<IFile> file)
{
    return std::make_shared<BatchOperation>(std::move(params), std::move(buffer), std::move(file));
}

BatchContext::BatchContext(unsigned _capacity) : capacity{_capacity}
{
    if (_capacity == 0) {
        throw std::invalid_argument("Batch capacity cannot be zero");
    }
    if (_capacity > MAX_SIZE) {
        throw std::invalid_argument("Batch capacity is limited to " + std::to_string(MAX_SIZE));
    }

    task_group = Context<IThreadPool>::get()->makeTaskGroup();
}

BatchContext::~BatchContext() = default;

unsigned
BatchContext::get_capacity() const noexcept
{
    return capacity;
}

void
BatchContext::submit_operations(const hipFileIOParams_t *params, unsigned num_params,
                                IBatchOperationFactory *operation_factory)
{
    std::unique_lock<std::shared_mutex> _ulock{context_mutex};

    // Check num_params first before doing anything else
    if (num_params > capacity - outstanding_ops.size()) {
        throw BatchFull();
    }

    std::vector<std::shared_ptr<IBatchOperation>> pending_ops{};
    BatchOperationFactory                         default_factory{};
    IBatchOperationFactory                       &factory =
        operation_factory == nullptr ? default_factory : *operation_factory;

    // It would be more performant to be able to perform multiple lookups
    // rather than waiting to lock the DriverState lock for each lookup.
    for (unsigned i = 0; i < num_params; i++) {
        // Make a copy of the params so another thread cannot modify the operation.
        auto param_copy = std::make_unique<const hipFileIOParams_t>(params[i]);
        // flags currently unused. Ambiguous if flags in hipFileBatchIOSubmit is for buffer or
        // file flags.
        auto [_file, _buffer] =
            Context<DriverState>::get()->getFileAndBuffer(param_copy->fh, param_copy->u.batch.devPtr_base);
        auto op = factory.create(std::move(param_copy), std::move(_buffer), std::move(_file));

        pending_ops.push_back(std::move(op));
    }

    // All submitted operations look valid at this point. Accept them.
    for (const auto &op : pending_ops) {
        op->mark_pending();
    }
    outstanding_ops.insert(pending_ops.begin(), pending_ops.end());

    auto self = shared_from_this();
    for (const auto &op : pending_ops) {
        task_group->run([self, op]() {
            try {
                op->run();
            }
            catch (...) {
                op->record_internal_error();
            }
            // Briefly serialize with any waiter mid-predicate-evaluation so notify_all is not
            // delivered before the waiter has actually entered wait(). See condition_variable
            // missed-wakeup pattern: state is mutated under per-op state_mutex, not the
            // context_mutex the waiter passes to wait(), so context_mutex is needed here.
            {
                std::shared_lock<std::shared_mutex> _serialize{self->context_mutex};
            }
            self->status_cv.notify_all();
        });
    }
}

void
BatchContext::get_status(unsigned min_nr, unsigned *nr, hipFileIOEvents_t *iocbp, struct timespec *timeout)
{
    if (nr == nullptr) {
        throw std::invalid_argument("Number of events cannot be null");
    }
    if (*nr > 0 && iocbp == nullptr) {
        throw std::invalid_argument("Event buffer cannot be null");
    }
    if (min_nr > *nr) {
        throw std::invalid_argument("Minimum event count exceeds event buffer capacity");
    }
    validate_timeout(timeout);

    const unsigned event_capacity = *nr;
    *nr                           = 0;

    std::unique_lock<std::shared_mutex> lock{context_mutex};

    auto terminal_count = [this]() {
        unsigned count = 0;
        for (const auto &op : outstanding_ops) {
            if (op->is_terminal()) {
                count++;
            }
        }
        return count;
    };

    auto collect_terminal_events = [this, event_capacity, nr, iocbp]() {
        unsigned copied = 0;
        for (auto op_iter = outstanding_ops.begin();
             op_iter != outstanding_ops.end() && copied < event_capacity;) {
            if (!(*op_iter)->is_terminal()) {
                ++op_iter;
                continue;
            }

            iocbp[copied++] = (*op_iter)->event();
            op_iter         = outstanding_ops.erase(op_iter);
        }
        *nr = copied;
        return copied;
    };

    if (outstanding_ops.empty() || event_capacity == 0) {
        return;
    }

    // Cap to what's actually outstanding so an over-large min_nr does not block forever.
    min_nr = std::min(min_nr, static_cast<unsigned>(outstanding_ops.size()));

    if (min_nr == 0 || terminal_count() >= min_nr || is_zero_timeout(timeout)) {
        collect_terminal_events();
        return;
    }

    // The second clause guards against a concurrent get_status() peer collecting terminal
    // events out from under us and dropping outstanding_ops below the cap captured at entry —
    // without it this waiter would block until timeout even though min_nr can no longer be met.
    auto ready = [&terminal_count, min_nr, this]() {
        return terminal_count() >= min_nr || outstanding_ops.size() < min_nr || outstanding_ops.empty();
    };

    if (timeout == nullptr) {
        status_cv.wait(lock, ready);
    }
    else {
        status_cv.wait_until(lock, timeout_deadline(timeout), ready);
    }

    collect_terminal_events();
}

void
BatchContextMap::clear()
{
    std::unique_lock<std::shared_mutex> ulock{batch_mutex};
    active_contexts.clear();
}

hipFileBatchHandle_t
BatchContextMap::createContext(unsigned capacity)
{
    auto                 context = std::shared_ptr<IBatchContext>{new BatchContext{capacity}};
    hipFileBatchHandle_t handle  = context.get();

    // Should not need to worry about duplicate keys unless the application
    // somehow deallocates this handle...

    std::unique_lock<std::shared_mutex> ulock{batch_mutex};
    active_contexts[handle] = std::move(context);
    return handle;
}

void
BatchContextMap::destroyContext(hipFileBatchHandle_t handle)
{
    std::unique_lock<std::shared_mutex> ulock{batch_mutex};

    auto context = active_contexts.find(handle);
    if (context == active_contexts.end()) {
        throw InvalidBatchHandle();
    }
    // TODO: Check for outstanding operations.
    // TODO: Attempt to cancel any outstanding operations.
    // TODO: Determine if we return unconditionally or require
    //       outstanding ops to terminate first.
    active_contexts.erase(handle);
}

std::shared_ptr<IBatchContext>
BatchContextMap::get(hipFileBatchHandle_t handle)
{
    // NOTE: This mutex only protects the map, so we'll
    //       also need to protect the data
    std::shared_lock<std::shared_mutex> slock{batch_mutex};

    auto context = active_contexts.find(handle);
    if (context == active_contexts.end()) {
        throw InvalidBatchHandle();
    }
    return context->second;
}

}
