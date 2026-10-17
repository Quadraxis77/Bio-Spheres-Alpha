//! System CPU busy time. Warm-up and failed samples are never published as load.

use std::time::{Duration, Instant};

pub(super) struct CpuSample {
    pub total: f32,
    pub per_core: Vec<f32>,
}

pub(super) struct CpuUsageSampler {
    last_refresh: Instant,
    core_count: usize,
    #[cfg(windows)]
    query: Option<windows::CpuQuery>,
}

impl CpuUsageSampler {
    pub fn new(_system: &mut sysinfo::System) -> Self {
        #[cfg(windows)]
        let (core_count, query) = {
            let core_count = windows::logical_cpu_count();
            (core_count, windows::CpuQuery::new())
        };
        #[cfg(not(windows))]
        let core_count = {
            _system.refresh_cpu_usage(); // Establish the baseline; discard its usage.
            _system.cpus().len()
        };
        Self {
            last_refresh: Instant::now(),
            core_count,
            #[cfg(windows)]
            query,
        }
    }

    pub fn core_count(&self) -> usize {
        self.core_count
    }

    fn sample_due(&mut self, now: Instant) -> bool {
        // Use actual elapsed time, independent of render dt or repeated refreshes.
        if now.saturating_duration_since(self.last_refresh)
            < Duration::from_secs(1).max(sysinfo::MINIMUM_CPU_UPDATE_INTERVAL)
        {
            return false;
        }
        self.last_refresh = now;
        true
    }

    pub fn sample(&mut self, _system: &mut sysinfo::System) -> Option<CpuSample> {
        if !self.sample_due(Instant::now()) {
            return None;
        }
        #[cfg(windows)]
        {
            if let Some(query) = &self.query {
                query.sample()
            } else {
                // Retry unavailable counters, then wait for their second sample.
                self.query = windows::CpuQuery::new();
                None
            }
        }
        #[cfg(not(windows))]
        {
            _system.refresh_cpu_usage();
            Some(CpuSample {
                total: _system.global_cpu_usage(),
                per_core: _system.cpus().iter().map(|cpu| cpu.cpu_usage()).collect(),
            })
        }
    }
}

#[cfg(windows)]
mod windows {
    use super::CpuSample;
    use windows_sys::Win32::System::{
        Performance::{
            PdhAddEnglishCounterW, PdhCloseQuery, PdhCollectQueryData, PdhGetFormattedCounterValue,
            PdhOpenQueryW, PDH_CSTATUS_NEW_DATA, PDH_CSTATUS_VALID_DATA, PDH_FMT_COUNTERVALUE,
            PDH_FMT_DOUBLE,
        },
        Threading::{GetActiveProcessorCount, GetActiveProcessorGroupCount},
    };

    pub fn logical_cpu_count() -> usize {
        unsafe { GetActiveProcessorCount(u16::MAX) as usize }
    }

    pub struct CpuQuery {
        handle: isize,
        counters: Vec<isize>, // Total first, then one per logical CPU.
    }

    impl CpuQuery {
        pub fn new() -> Option<Self> {
            let mut handle = 0;
            // PDH owns the returned handles; Drop closes the query and counters.
            if unsafe { PdhOpenQueryW(std::ptr::null(), 0, &mut handle) } != 0 {
                return None;
            }
            let mut query = Self {
                handle,
                counters: Vec::new(),
            };
            query.add_counter("_Total")?;
            for group in 0..unsafe { GetActiveProcessorGroupCount() } {
                for cpu in 0..unsafe { GetActiveProcessorCount(group) } {
                    query.add_counter(&format!("{group},{cpu}"))?;
                }
            }
            // Rate counters need two observations separated by time.
            if unsafe { PdhCollectQueryData(handle) } != 0 {
                return None;
            }
            Some(query)
        }

        fn add_counter(&mut self, instance: &str) -> Option<()> {
            let path: Vec<u16> = format!(r"\Processor Information({instance})\% Processor Time")
                .encode_utf16()
                .chain(Some(0))
                .collect();
            let mut counter = 0;
            if unsafe { PdhAddEnglishCounterW(self.handle, path.as_ptr(), 0, &mut counter) } != 0 {
                return None;
            }
            self.counters.push(counter);
            Some(())
        }

        pub fn sample(&self) -> Option<CpuSample> {
            if unsafe { PdhCollectQueryData(self.handle) } != 0 {
                return None;
            }
            let mut values = Vec::with_capacity(self.counters.len());
            for &counter in &self.counters {
                let mut value: PDH_FMT_COUNTERVALUE = unsafe { std::mem::zeroed() };
                let status = unsafe {
                    PdhGetFormattedCounterValue(
                        counter,
                        PDH_FMT_DOUBLE,
                        std::ptr::null_mut(),
                        &mut value,
                    )
                };
                values.push(validated_usage(status, value.CStatus, unsafe {
                    value.Anonymous.doubleValue
                })?);
            }
            Some(CpuSample {
                total: values[0],
                per_core: values[1..].to_vec(),
            })
        }
    }

    fn validated_usage(call_status: u32, data_status: u32, value: f64) -> Option<f32> {
        // A successful API call does not necessarily mean its counter data is valid.
        (call_status == 0
            && matches!(data_status, PDH_CSTATUS_VALID_DATA | PDH_CSTATUS_NEW_DATA)
            && value.is_finite()
            && (0.0..=100.0).contains(&value))
        .then_some(value as f32)
    }

    impl Drop for CpuQuery {
        fn drop(&mut self) {
            unsafe {
                PdhCloseQuery(self.handle);
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn invalid_counter_data_is_not_published_as_cpu_load() {
            assert_eq!(validated_usage(0xC0000BBA, 0, 0.0), None);
            assert_eq!(validated_usage(0, 0xC0000BBA, 95.0), None);
            assert_eq!(validated_usage(0, 0, f64::NAN), None);
            assert_eq!(validated_usage(0, 0, 101.0), None);
            assert_eq!(validated_usage(0, 1, 95.0), Some(95.0));
            assert_eq!(validated_usage(0, 0, 100.0), Some(100.0));
            assert_eq!(validated_usage(0, 0, 0.0), Some(0.0));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn startup_and_back_to_back_refreshes_wait_for_a_full_sample_interval() {
        let mut system = sysinfo::System::new();
        let mut sampler = CpuUsageSampler::new(&mut system);
        let start = sampler.last_refresh;
        assert!(!sampler.sample_due(start));
        assert!(!sampler.sample_due(start + Duration::from_millis(199)));
        assert!(!sampler.sample_due(start + Duration::from_millis(999)));
        assert!(sampler.sample_due(start + Duration::from_secs(1)));
        assert!(!sampler.sample_due(start + Duration::from_secs(1)));
        assert!(sampler.sample_due(start + Duration::from_secs(2)));
    }
}
