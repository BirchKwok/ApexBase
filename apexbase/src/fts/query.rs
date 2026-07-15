fn phrase_query(query: &str) -> (&str, bool) {
    let query = query.trim();
    if query.len() >= 2 && query.starts_with('"') && query.ends_with('"') {
        (&query[1..query.len() - 1], true)
    } else {
        (query, false)
    }
}

fn matching_docs(state: &IndexState, analyzed: &AnalyzedDocument, phrase: bool) -> FtsResult<RoaringTreemap> {
    let mut postings: Vec<RoaringTreemap> = analyzed
        .terms
        .iter()
        .map(|term| state.posting(term))
        .collect::<FtsResult<Vec<_>>>()?;
    postings.sort_unstable_by_key(RoaringTreemap::len);
    let mut postings = postings.into_iter();
    let Some(mut result) = postings.next() else {
        return Ok(RoaringTreemap::new());
    };
    for posting in postings {
        result &= posting;
        if result.is_empty() {
            return Ok(result);
        }
    }
    if !phrase {
        return Ok(result);
    }

    let query_tokens: Vec<u32> = analyzed
        .tokens
        .iter()
        .filter_map(|&local_id| {
            if local_id == 0 {
                None
            } else {
                analyzed
                    .terms
                    .get(local_id as usize - 1)
                    .and_then(|term| state.term_id(term))
            }
        })
        .collect();
    if query_tokens.is_empty() {
        return Ok(RoaringTreemap::new());
    }
    let mut phrase_matches = RoaringTreemap::new();
    for doc_id in result.iter() {
        if state.tokens_for_doc(doc_id).is_some_and(|tokens| {
            tokens
                .windows(query_tokens.len())
                .any(|w| w == query_tokens)
        }) {
            phrase_matches.insert(doc_id);
        }
    }
    Ok(phrase_matches)
}
