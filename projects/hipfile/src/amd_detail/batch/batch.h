/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "hipfile.h"

#include <condition_variable>
#include <cstddef>
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
class IBatchOperation {
public:
    virtual ~IBatchOperation() = default;

    /// @brief Mark the operation as accepted and ready to run.
    virtual void mark_pending() = 0;

    /// @brief Cancel the operation if it can be transitioned to Canceled; otherwise no-op.
    virtual void try_cancel() = 0;

    /// @brief Execute the operation.
    virtual void run() = 0;

    /// @brief Record an internal execution failure on the operation.
    virtual void record_internal_error() = 0;

    /// @brief Return a snapshot of the operation event state.
    virtual hipFileIOEvents_t event() const = 0;

    /// @brief Return whether the operation has reached a terminal status.
    virtual bool is_terminal() const = 0;
};

/// @brief Represents a single IO Request
class BatchOperation : public IBatchOperation {
public:
    /// @brief Internal operation status. Includes Running, which is not exposed via
    ///        the public API; get_status() and event() translate it to hipFilePending.
    enum class InternalStatus {
        Waiting,
        Pending,
        Running,
        Complete,
        Canceled,
        Invalid,
        Timeout,
        Failed,
    };

    ~BatchOperation() override = default;

    /// @brief Create an operation to handle and track an IO request.
    /// @param [in] params IO parameters
    /// @param [in] buffer Buffer corresponding to params->u.batch.devPtr_base
    /// @param [in] file File corresponding params->fh
    BatchOperation(std::unique_ptr<const hipFileIOParams_t> params, std::shared_ptr<IBuffer> buffer,
                   std::shared_ptr<IFile> file);

    /// @brief Mark the operation as accepted and ready to run.
    void mark_pending() override;

    /// @brief Cancel the operation if it can be transitioned to Canceled; otherwise no-op.
    void try_cancel() override;

    /// @brief Execute the operation.
    void run() override;

    /// @brief Record an internal execution failure on the operation.
    void record_internal_error() override;

    /// @brief Return a snapshot of the operation event state.
    hipFileIOEvents_t event() const override;

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

class IBatchOperationFactory {
public:
    virtual ~IBatchOperationFactory() = default;

    virtual std::shared_ptr<IBatchOperation> create(std::unique_ptr<const hipFileIOParams_t> params,
                                                    std::shared_ptr<IBuffer>                 buffer,
                                                    std::shared_ptr<IFile>                   file) = 0;
};

class BatchOperationFactory : public IBatchOperationFactory {
public:
    std::shared_ptr<IBatchOperation> create(std::unique_ptr<const hipFileIOParams_t> params,
                                            std::shared_ptr<IBuffer>                 buffer,
                                            std::shared_ptr<IFile>                   file) override;
};

class IBatchContext {
public:
    static constexpr unsigned MAX_SIZE = 128;

    virtual ~IBatchContext()                                                                = default;
    virtual unsigned get_capacity() const noexcept                                          = 0;
    virtual void     submit_operations(const hipFileIOParams_t *params, unsigned num_params,
                                       IBatchOperationFactory *operation_factory = nullptr) = 0;
    virtual void     get_status(unsigned min_nr, unsigned *nr, hipFileIOEvents_t *iocbp,
                                struct timespec *timeout)                                   = 0;
    virtual void     cancel_operations()                                                    = 0;
};

class BatchContext : public IBatchContext, public std::enable_shared_from_this<BatchContext> {
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
    void submit_operations(const hipFileIOParams_t *params, const unsigned num_params,
                           IBatchOperationFactory *operation_factory = nullptr) override;

    ///
    /// @brief Poll for completed operations from this Context.
    /// @param [in]     min_nr  Minimum number of events requested before returning.
    /// @param [in,out] nr      Input event capacity and output number of events returned.
    /// @param [out]    iocbp   Event output buffer.
    /// @param [in]     timeout Maximum amount of time to wait.
    ///
    void get_status(unsigned min_nr, unsigned *nr, hipFileIOEvents_t *iocbp,
                    struct timespec *timeout) override;

    ///
    /// @brief Cancel outstanding operations from this Context.
    ///
    void cancel_operations() override;

private:
    const unsigned capacity;

    /// Per-Context mutex to limit access to one caller at a time.
    /// Shared as internally we can be more strategic about concurrent access.
    mutable std::shared_mutex context_mutex;

    /// Wakes callers waiting for operations to become terminal.
    std::condition_variable_any status_cv;

    /// An outstanding operation is a BatchOperation that has been submitted
    /// but is not yet complete or completed but not yet retrieved by the
    /// application.
    /// shared_ptr as it may need to be passed to a backend.
    std::unordered_set<std::shared_ptr<IBatchOperation>> outstanding_ops;

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
