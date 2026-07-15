fn normalize(text: &str) -> String {
    if text.is_ascii() {
        text.to_ascii_lowercase()
    } else {
        text.nfkc().flat_map(char::to_lowercase).collect()
    }
}

fn is_cjk(c: char) -> bool {
    matches!(c,
        '\u{3400}'..='\u{4dbf}' |
        '\u{4e00}'..='\u{9fff}' |
        '\u{f900}'..='\u{faff}' |
        '\u{3040}'..='\u{309f}' |
        '\u{30a0}'..='\u{30ff}' |
        '\u{31f0}'..='\u{31ff}' |
        '\u{ac00}'..='\u{d7af}'
    )
}

fn primary_tokens(normalized: &str, config: &FtsConfig) -> Vec<String> {
    let mut tokens = Vec::new();
    for segment in normalized.split_word_bounds() {
        let mut non_cjk = String::new();
        let flush_non_cjk = |buffer: &mut String, output: &mut Vec<String>| {
            for word in buffer.unicode_words() {
                if word.chars().count() >= config.min_term_length {
                    output.push(word.to_string());
                }
            }
            buffer.clear();
        };
        for character in segment.chars() {
            if is_cjk(character) {
                flush_non_cjk(&mut non_cjk, &mut tokens);
                tokens.push(character.to_string());
            } else {
                non_cjk.push(character);
            }
        }
        flush_non_cjk(&mut non_cjk, &mut tokens);
    }
    tokens
}

#[derive(Default)]
struct AnalysisBuilder {
    terms: AHashMap<String, u32>,
    tokens: Vec<u32>,
}

impl AnalysisBuilder {
    fn intern(&mut self, term: String) -> u32 {
        if let Some(&term_id) = self.terms.get(&term) {
            return term_id;
        }
        let term_id = self.terms.len() as u32 + 1;
        self.terms.insert(term, term_id);
        term_id
    }

    fn add_text(&mut self, text: &str, config: &FtsConfig) {
        let normalized = normalize(text);
        if normalized.is_ascii() {
            for word in normalized
                .split(|character: char| !character.is_ascii_alphanumeric() && character != '_')
            {
                if word.len() >= config.min_term_length {
                    let term_id = self.intern(word.to_string());
                    self.tokens.push(term_id);
                }
            }
            return;
        }

        for token in primary_tokens(&normalized, config) {
            let term_id = self.intern(token);
            self.tokens.push(term_id);
        }
        let chars: Vec<char> = normalized.chars().collect();
        let mut start = 0;
        while start < chars.len() {
            if !is_cjk(chars[start]) {
                start += 1;
                continue;
            }
            let mut end = start + 1;
            while end < chars.len() && is_cjk(chars[end]) {
                end += 1;
            }
            let run = &chars[start..end];
            let max_n = config.max_chinese_length.max(1).min(run.len());
            for n in 1..=max_n {
                for offset in 0..=run.len() - n {
                    self.intern(run[offset..offset + n].iter().collect());
                }
            }
            start = end;
        }
    }

    fn finish(self) -> AnalyzedDocument {
        let mut entries: Vec<(String, u32)> = self.terms.into_iter().collect();
        entries.sort_unstable_by(|left, right| left.0.cmp(&right.0));
        let mut remap = vec![0; entries.len() + 1];
        let mut terms = Vec::with_capacity(entries.len());
        for (new_index, (term, old_id)) in entries.into_iter().enumerate() {
            remap[old_id as usize] = new_index as u32 + 1;
            terms.push(term);
        }
        let tokens = self
            .tokens
            .into_iter()
            .map(|term_id| {
                if term_id == 0 {
                    0
                } else {
                    remap[term_id as usize]
                }
            })
            .collect();
        AnalyzedDocument { terms, tokens }
    }
}

fn analyze_document(text: &str, config: &FtsConfig) -> AnalyzedDocument {
    let mut builder = AnalysisBuilder::default();
    builder.add_text(text, config);
    builder.finish()
}

fn analyze_values<'a>(
    values: impl IntoIterator<Item = &'a str>,
    config: &FtsConfig,
) -> AnalyzedDocument {
    let mut builder = AnalysisBuilder::default();
    for value in values {
        let prior_tokens = builder.tokens.len();
        if prior_tokens > 0 {
            builder.tokens.push(0);
        }
        let boundary_index = builder.tokens.len();
        builder.add_text(value, config);
        if builder.tokens.len() == boundary_index && prior_tokens > 0 {
            builder.tokens.pop();
        }
    }
    builder.finish()
}

fn analyze(text: &str, config: &FtsConfig) -> Vec<String> {
    analyze_document(text, config).terms
}

fn levenshtein(left: &str, right: &str, max_distance: usize) -> Option<usize> {
    let left: Vec<char> = left.chars().collect();
    let right: Vec<char> = right.chars().collect();
    if left.len().abs_diff(right.len()) > max_distance {
        return None;
    }
    let mut previous: Vec<usize> = (0..=right.len()).collect();
    let mut current = vec![0; right.len() + 1];
    for (i, &a) in left.iter().enumerate() {
        current[0] = i + 1;
        let mut row_min = current[0];
        for (j, &b) in right.iter().enumerate() {
            current[j + 1] = (previous[j + 1] + 1)
                .min(current[j] + 1)
                .min(previous[j] + usize::from(a != b));
            row_min = row_min.min(current[j + 1]);
        }
        if row_min > max_distance {
            return None;
        }
        std::mem::swap(&mut previous, &mut current);
    }
    (previous[right.len()] <= max_distance).then_some(previous[right.len()])
}

