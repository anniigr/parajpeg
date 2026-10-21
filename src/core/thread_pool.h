// Minimal thread pool with a parallel-for over an index range (dynamic chunking).
#pragma once
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace parajpeg::detail {

class ThreadPool {
public:
    explicit ThreadPool(int threads) {
        if (threads < 1) threads = 1;
        nthreads_ = threads;
        for (int i = 1; i < threads; ++i) workers_.emplace_back([this, i] { worker_loop(i); });
    }
    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lk(m_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& t : workers_) t.join();
    }
    int size() const { return nthreads_; }

    // Runs fn(i, thread_id) for i in [0, n). Blocks until done. Work items are claimed dynamically.
    void parallel_for(uint32_t n, const std::function<void(uint32_t, int)>& fn) {
        if (n == 0) return;
        if (nthreads_ == 1 || n == 1) {
            for (uint32_t i = 0; i < n; ++i) fn(i, 0);
            return;
        }
        Job job;
        job.n = n;
        job.fn = &fn;
        {
            std::lock_guard<std::mutex> lk(m_);
            job_ = &job;
            ++generation_;
        }
        cv_.notify_all();
        run_job(job, 0);
        // Wait for all workers to finish the job
        std::unique_lock<std::mutex> lk(m_);
        done_cv_.wait(lk, [&] { return job.finished_workers.load() == nthreads_ - 1 && job.active.load() == 0; });
        job_ = nullptr;
    }

private:
    struct Job {
        uint32_t n = 0;
        const std::function<void(uint32_t, int)>* fn = nullptr;
        std::atomic<uint32_t> next{0};
        std::atomic<int> active{0};
        std::atomic<int> finished_workers{0};
    };

    void run_job(Job& job, int tid) {
        job.active.fetch_add(1);
        for (;;) {
            uint32_t i = job.next.fetch_add(1);
            if (i >= job.n) break;
            (*job.fn)(i, tid);
        }
        job.active.fetch_sub(1);
    }

    void worker_loop(int tid) {
        uint64_t seen = 0;
        for (;;) {
            Job* job = nullptr;
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait(lk, [&] { return stop_ || generation_ != seen; });
                if (stop_) return;
                seen = generation_;
                job = job_;
            }
            if (job) {
                run_job(*job, tid);
                job->finished_workers.fetch_add(1);
                {
                    std::lock_guard<std::mutex> lk(m_);
                }
                done_cv_.notify_one();
            }
        }
    }

    int nthreads_ = 1;
    std::vector<std::thread> workers_;
    std::mutex m_;
    std::condition_variable cv_, done_cv_;
    Job* job_ = nullptr;
    uint64_t generation_ = 0;
    bool stop_ = false;
};

}  // namespace parajpeg::detail
