/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "hipfile.h"

#include <memory>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace hipFile {
class IBuffer;
}
namespace hipFile {
class IFile;
}
namespace hipFile {
class ITaskGroup;
}

namespace hipFile {

struct InvalidBatchHandle : public std::invalid_argument {
    InvalidBatchHandle() : std::invalid_argument{"Invalid batch handle"}
    {
    }
};

struct BatchFull : public std::invalid_argument {
    BatchFull() : std::invalid_argument{"Not enough room in batch"}
    {
    }
};

struct InvalidStateTransition : public std::logic_error {
    InvalidStateTransition(hipFileStatus_t from, hipFileStatus_t to);
    InvalidStateTransition(const char *from, const char *to);
};

/// @brief Represents a single IO Request
class BatchOperation {
public:
    /// @brief Create an operation to handle and track an IO request.
    /// @param [in] params IO parameters
    /// @param [in] buffer Buffer corresponding to params->u.batch.devPtr_base
    /// @param [in] file File corresponding params->fh
    BatchOperation(std::unique_ptr<const hipFileIOParams_t> params, std::shared_ptr<IBuffer> buffer,
                   std::shared_ptr<IFile> file);

    /// @brief Mark the operation as accepted and ready to run.
    void mark_pending();

    /// @brief Cancel the operation if it is pending.
    void cancel();

    /// @brief Execute the operation.
    void run();

    /// @brief Record an internal execution failure on the operation.
    void record_internal_error();

    /// @brief Return the current operation status.
    hipFileStatus_t get_status() const;

    /// @brief Return the operation result.
    ssize_t get_result() const;

    /// @brief Return whether the operation has reached a terminal status.
    bool is_terminal() const override;

private:
    /// @brief A copy of the params provided by the application.
    /// @internal Keep this listed at the top of BatchOperation.
    const std::unique_ptr<const hipFileIOParams_t> io_params;

    /// @brief A reference to the specified Buffer.
    const std::shared_ptr<const IBuffer> buffer;

    /// @brief A reference to the specified registered File.
    const std::shared_ptr<const IFile> file;

    /// @brief Protects status and ret.
    mutable std::mutex state_mutex;

    /// @brief Current operation status.
    InternalStatus status{InternalStatus::Waiting};

    /// @brief Result returned by hipFileRead or hipFileWrite.
    ssize_t ret{0};

    /// @brief Move to the next operation status. Caller must hold state_mutex.
    void transition_to(InternalStatus next, ssize_t next_ret);

    /// @brief Move to the next operation status. Caller must hold state_mutex.
    void transition_to(InternalStatus next);

    /// @brief Return whether an internal status is terminal.
    static bool is_terminal_status(InternalStatus status) noexcept;
};

class IBatchContext {
public:
    static constexpr unsigned MAX_SIZE = 128;

    virtual ~IBatchContext()                                                                 = default;
    virtual unsigned get_capacity() const noexcept                                           = 0;
    virtual void     submit_operations(const hipFileIOParams_t *params, unsigned num_params) = 0;
};

class BatchContext : public IBatchContext {
public:
    ~BatchContext() override;

    ///
    /// @brief Return the max number of concurrent operations supported by this BatchContext.
    ///
    /// @return The max number of concurrent operations that can be processed by this BatchContext.
    /// @note This may not exceed the value returned by `MAX_SIZE`.
    unsigned get_capacity() const noexcept override;

    ///
    /// @brief Submit one or more operations to this Context.
    /// @param [in] params     Pointer to the operations to enqueue.
    /// @param [in] num_params Number of operations to enqueue.
    ///
    /// @note This is an All or None operation. If one submitted operation is not valid, no operations
    ///       will be submitted.
    ///
    void submit_operations(const hipFileIOParams_t *params, const unsigned num_params) override;

private:
    const unsigned capacity;

    /// Per-Context mutex to limit access to one caller at a time.
    /// Shared as internally we can be more strategic about concurrent access.
    mutable std::shared_mutex context_mutex;

    /// An outstanding operation is a BatchOperation that has been submitted
    /// but is not yet complete or completed but not yet retrieved by the
    /// application.
    /// shared_ptr as it may need to be passed to a backend.
    std::unordered_set<std::shared_ptr<BatchOperation>> outstanding_ops;

    /// Task group used for all submitted operations owned by this context.
    std::unique_ptr<ITaskGroup> task_group;

    BatchContext(unsigned capacity);

    friend class BatchContextMap;
};

class BatchContextMap {
public:
    /*!
     * @brief Create a new batch context
     * @param capacity Maximum number of outstanding operations that this context can manage
     * @return An opaque handle used to reference this new batch context
     */
    hipFileBatchHandle_t createContext(unsigned capacity);

    /*!
     * @brief Destroy a batch context and release all associated resources
     * @param handle The handle for the batch context to destroy
     */
    void destroyContext(hipFileBatchHandle_t handle);

    /*!
     * @brief Get a batch context
     * @param handle The opaque handle associated with a batch context
     * @return A batch context
     */
    std::shared_ptr<IBatchContext> get(hipFileBatchHandle_t handle);

    /*!
     * @brief Clear the contents
     */
    void clear();

private:
    /// batch context lookup table
    std::unordered_map<hipFileBatchHandle_t, std::shared_ptr<IBatchContext>> active_contexts;

    /// Mutex to protect the active context map
    mutable std::shared_mutex batch_mutex;
};

}
