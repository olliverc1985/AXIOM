//! SMS-specific feature extraction for phishing detection.
//!
//! Extracts hand-crafted signals that are strong indicators of SMS phishing,
//! layered on top of AXIOM's general structural encoder. These features capture
//! patterns specific to smishing (SMS phishing) attacks commonly seen on UK
//! mobile networks.

/// SMS-specific feature extractor.
///
/// Produces a fixed-size feature vector of SMS phishing indicators that
/// complement the general structural encoder.
pub struct SmsFeatureExtractor;

/// SMS-specific features extracted from a message.
#[derive(Debug, Clone)]
pub struct SmsFeatures {
    /// Number of URLs in the message.
    pub url_count: usize,
    /// Whether URLs use suspicious TLDs (.info, .xyz, .top, etc).
    pub suspicious_tld: bool,
    /// Whether the message contains urgency language.
    pub has_urgency: bool,
    /// Whether the message impersonates a known brand.
    pub brand_impersonation: bool,
    /// Ratio of uppercase characters to total alphabetic characters.
    pub caps_ratio: f32,
    /// Number of exclamation marks.
    pub exclamation_count: usize,
    /// Whether the message requests personal/financial information.
    pub requests_info: bool,
    /// Whether the message contains a call-to-action with a deadline.
    pub has_deadline: bool,
    /// Whether the message contains monetary amounts.
    pub has_monetary: bool,
    /// Whether the message uses textspeak/leetspeak (common in spam).
    pub has_textspeak: bool,
    /// Whether the message contains premium rate numbers.
    pub has_premium_number: bool,
    /// Message length in characters.
    pub message_length: usize,
}

impl SmsFeatureExtractor {
    /// Extract SMS-specific phishing features from a message.
    pub fn extract(text: &str) -> SmsFeatures {
        let lower = text.to_lowercase();

        SmsFeatures {
            url_count: Self::count_urls(text),
            suspicious_tld: Self::has_suspicious_tld(&lower),
            has_urgency: Self::has_urgency_language(&lower),
            brand_impersonation: Self::has_brand_impersonation(&lower),
            caps_ratio: Self::caps_ratio(text),
            exclamation_count: text.chars().filter(|&c| c == '!').count(),
            requests_info: Self::requests_information(&lower),
            has_deadline: Self::has_deadline_language(&lower),
            has_monetary: Self::has_monetary_amounts(&lower),
            has_textspeak: Self::has_textspeak(&lower),
            has_premium_number: Self::has_premium_rate_number(text),
            message_length: text.len(),
        }
    }

    /// Compute a phishing risk score from 0.0 (safe) to 1.0 (definitely phishing).
    pub fn risk_score(features: &SmsFeatures) -> f32 {
        let mut score = 0.0f32;

        // URLs are the strongest single signal
        if features.url_count > 0 {
            score += 0.15;
        }
        if features.url_count > 1 {
            score += 0.10;
        }
        if features.suspicious_tld {
            score += 0.15;
        }

        // Urgency + brand impersonation is classic phishing
        if features.has_urgency {
            score += 0.12;
        }
        if features.brand_impersonation {
            score += 0.12;
        }
        if features.has_urgency && features.brand_impersonation {
            score += 0.08; // combined signal bonus
        }

        // Requesting info with a URL is high risk
        if features.requests_info {
            score += 0.10;
        }
        if features.requests_info && features.url_count > 0 {
            score += 0.05;
        }

        // Deadline pressure
        if features.has_deadline {
            score += 0.08;
        }

        // Caps and exclamation abuse
        if features.caps_ratio > 0.3 {
            score += 0.05;
        }
        if features.exclamation_count >= 3 {
            score += 0.05;
        }

        // Monetary amounts (prizes, refunds)
        if features.has_monetary {
            score += 0.05;
        }

        // Premium rate numbers
        if features.has_premium_number {
            score += 0.15;
        }

        // Textspeak in commercial context is spammy
        if features.has_textspeak && (features.has_monetary || features.url_count > 0) {
            score += 0.05;
        }

        score.min(1.0)
    }

    fn count_urls(text: &str) -> usize {
        let lower = text.to_lowercase();
        lower.matches("http://").count()
            + lower.matches("https://").count()
            + lower.matches("www.").count()
    }

    fn has_suspicious_tld(lower: &str) -> bool {
        let suspicious = [
            ".info", ".xyz", ".top", ".click", ".link", ".buzz", ".live",
            ".online", ".site", ".club", ".work", ".icu", ".gq", ".ml",
            ".tk", ".cf", ".ga",
        ];
        suspicious.iter().any(|tld| lower.contains(tld))
    }

    fn has_urgency_language(lower: &str) -> bool {
        let urgency = [
            "urgent", "immediately", "act now", "right away", "expires today",
            "final warning", "last chance", "don't delay", "time is running out",
            "within 24 hours", "within 48 hours", "account will be",
            "will be suspended", "will be closed", "will be blocked",
            "will be revoked", "unless you", "before it expires",
        ];
        urgency.iter().any(|phrase| lower.contains(phrase))
    }

    fn has_brand_impersonation(lower: &str) -> bool {
        let brands = [
            "royal mail", "hmrc", "amazon", "netflix", "apple", "paypal",
            "hsbc", "barclays", "lloyds", "natwest", "santander", "tsb",
            "dvla", "nhs", "bt ", "ee ", "o2 ", "vodafone", "three ",
            "sky ", "tesco", "argos", "dpd", "hermes", "evri",
            "whatsapp",
        ];
        // Only flag as impersonation if combined with a URL or urgency
        let has_brand = brands.iter().any(|brand| lower.contains(brand));
        let has_url = lower.contains("http") || lower.contains("www.");
        has_brand && has_url
    }

    fn caps_ratio(text: &str) -> f32 {
        let alpha: Vec<char> = text.chars().filter(|c| c.is_alphabetic()).collect();
        if alpha.is_empty() {
            return 0.0;
        }
        let upper = alpha.iter().filter(|c| c.is_uppercase()).count();
        upper as f32 / alpha.len() as f32
    }

    fn requests_information(lower: &str) -> bool {
        let phrases = [
            "verify your", "confirm your", "update your", "enter your",
            "provide your", "submit your", "send your", "bank details",
            "card details", "personal details", "identity", "password",
            "login", "log in", "sign in", "credentials",
        ];
        phrases.iter().any(|p| lower.contains(p))
    }

    fn has_deadline_language(lower: &str) -> bool {
        let deadlines = [
            "24 hours", "48 hours", "expires", "expiring", "today only",
            "limited time", "before midnight", "ends today", "overdue",
            "past due",
        ];
        deadlines.iter().any(|d| lower.contains(d))
    }

    fn has_monetary_amounts(lower: &str) -> bool {
        // Match £ or $ followed by digits, or "pounds"/"dollars"
        lower.contains('£')
            || lower.contains('$')
            || lower.contains("pounds")
            || lower.contains("refund")
            || lower.contains("prize")
            || lower.contains("won ")
            || lower.contains("winner")
            || lower.contains("cashback")
    }

    fn has_textspeak(lower: &str) -> bool {
        let patterns = [
            " ur ", " u ", " 4 ", " 2 ", " txt ", " msg ",
            "congratz", "congrats!", "plz", "thx",
        ];
        patterns.iter().any(|p| lower.contains(p))
    }

    fn has_premium_rate_number(text: &str) -> bool {
        // UK premium rate numbers: 09xx, 087x
        let patterns = ["090", "091", "0871", "0872", "0873"];
        patterns.iter().any(|p| text.contains(p))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clean_message() {
        let features = SmsFeatureExtractor::extract("Hi mate, fancy a pint after work?");
        assert_eq!(features.url_count, 0);
        assert!(!features.has_urgency);
        assert!(!features.brand_impersonation);
        assert!(SmsFeatureExtractor::risk_score(&features) < 0.2);
    }

    #[test]
    fn test_phishing_message() {
        let features = SmsFeatureExtractor::extract(
            "URGENT: Your HSBC account will be suspended! Verify now: http://hsbc-verify.info"
        );
        assert!(features.url_count > 0);
        assert!(features.has_urgency);
        assert!(features.suspicious_tld);
        assert!(SmsFeatureExtractor::risk_score(&features) > 0.5);
    }

    #[test]
    fn test_caps_ratio() {
        assert!(SmsFeatureExtractor::extract("HELLO WORLD").caps_ratio > 0.9);
        assert!(SmsFeatureExtractor::extract("hello world").caps_ratio < 0.01);
    }

    #[test]
    fn test_premium_number() {
        let features = SmsFeatureExtractor::extract("Call 09061234567 to claim your prize!");
        assert!(features.has_premium_number);
    }

    #[test]
    fn test_monetary() {
        let features = SmsFeatureExtractor::extract("You won £500! Claim now!");
        assert!(features.has_monetary);
    }
}
