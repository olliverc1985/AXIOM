//! SMS Phishing Detection — classify SMS messages for gateway routing.
//!
//! Uses AXIOM's dual-encoder architecture (structural + semantic) to classify
//! inbound SMS messages into three routing decisions:
//!
//! - **Gateway** — clean message, route to mobile network
//! - **HoldingPen** — suspicious, held for human review
//! - **Blocked** — definite phishing, reject outright
//!
//! # Architecture
//!
//! SMS messages pass through two encoders in parallel:
//! 1. **Structural encoder** (128-dim) — hand-crafted features tuned for SMS:
//!    URL density, urgency markers, ALL CAPS ratio, punctuation abuse, sender patterns
//! 2. **Semantic encoder** (128-dim) — transformer embedding of message meaning
//!
//! The 256-dim concatenated vector is classified by a linear head with softmax.
//!
//! # Usage
//!
//! ```no_run
//! use axiom::sms::{SmsPhishingDetector, SmsRoutingDecision, SmsVerdict};
//!
//! // Load a trained detector
//! let detector = SmsPhishingDetector::load("sms_phishing_weights.json");
//!
//! // Classify an SMS
//! let decision = detector.classify("Your parcel is waiting. Confirm: http://dodgy-link.com");
//! match decision.verdict {
//!     SmsVerdict::Gateway => println!("Route to network"),
//!     SmsVerdict::HoldingPen => println!("Hold for review"),
//!     SmsVerdict::Blocked => println!("Reject — phishing"),
//! }
//! ```

pub mod detector;
pub mod features;

pub use detector::{SmsPhishingDetector, SmsPhishingWeights, SmsRoutingDecision, SmsVerdict};
pub use features::SmsFeatureExtractor;
