/* confidence-render.js — shared "Thermo view" renderer for the dev testers.
 *
 * Built once and <script src>'d from all four testers (model_tester_demo.html,
 * legal_chat_demo.html, legal_uk_chat_demo.html, intake_chat_demo.html) rather than pasted
 * into each, specifically so a threshold only lives in one place. Four copies of this logic
 * is exactly how the Thermo Chat reference's own slider got out of sync with itself (its
 * initial HTML slider values are 0.45/0.70/0.85/1.00, while its Reset button and its own JS
 * module variables use 0.66/0.76/0.88/1.00 — a bug born from duplicated constants).
 *
 * Consumes the {"type":"confidence", ...} envelope and (where a tester opts in via
 * confidence_live: true on its chat payload) the live [CONF]{...}[/CONF] per-token frames —
 * see chat-wonder-v2-api's confidence.py for the exact schema both are built from.
 *
 * RENDER STRATEGY (second major revision — read this before changing render()):
 * Render markdown via the caller's `mdParse` (marked.parse) FIRST, unconditionally, THEN
 * overlay spans/words by walking the RESULT'S DOM TEXT NODES and wrapping matches in <mark>
 * elements in place. The first version of this file matched spans against the raw markdown
 * SOURCE STRING and rendered that escaped string directly whenever any span existed at all —
 * which meant the moment a single span existed, the whole message fell back to literal
 * "## heading" / "**bold**" / "- bullet" syntax, discarding markdown structure for the ENTIRE
 * message, not just the matched part. Operating on the already-parsed DOM instead means
 * headings render as headings, bullets as bullets, and a citation link stays a real <a>
 * element — exactly the model the app's own assistant-message.tsx already uses for citation
 * pills (buildComponents decorates markdown-rendered output, never raw source).
 *
 * This also makes confidence.py's span `quote` fields CLEANED markdown-rendered text (not
 * verbatim raw source) — a deliberate, matching change on the server side (see
 * confidence.py's _plain_for_match), because a verbatim "## Summary" string will never appear
 * in rendered DOM text (marked.js strips the "##" when building the <h2>).
 */
(function () {
  "use strict";

  // Three categories, matching confidence.py's provenance_spans formula exactly: a sentence is
  // either quoted/near-verbatim from an identified source, the model's own conclusion drawing
  // on an identified source, or the model's own reasoning with no source behind it at all.
  var TIER_STYLE = {
    QUOTED: { bg: "rgba(34,197,94,0.14)", fg: "#16a34a", label: "Quoted from source" },
    SOURCED: { bg: "rgba(234,179,8,0.16)", fg: "#a16207", label: "Conclusion from a source" },
    UNSOURCED: { bg: "rgba(148,163,184,0.16)", fg: "#64748b", label: "No source" },
  };

  // hn = h / ln(5), so these are fractions of the maximum possible top-5 entropy, not raw
  // nats. STARTING GUESSES ONLY — the Thermo Chat reference's own measured mean hn on a real
  // model was far below its own displayed thresholds, so these must be replaced with real
  // percentiles once CONFIDENCE_SPANS has run against representative traffic (see the
  // architecture doc's "Calibration" section).
  var ENTROPY_BANDS = { blue: 0.25, orange: 0.45, red: 0.65 };
  var ENTROPY_PURPLE_HN = 0.9; // fraction of ln5 above which the purple "high uncertainty" background shows

  function entropyColor(hn) {
    if (hn == null) return null;
    if (hn >= ENTROPY_BANDS.red) return "#ef4444";
    if (hn >= ENTROPY_BANDS.orange) return "#fb923c";
    if (hn >= ENTROPY_BANDS.blue) return "#38bdf8";
    return null;
  }

  function escHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  // Fallback-only: used by _renderPlainOrMarkdown when no mdParse function is available (a
  // tester that never rendered markdown in the first place) or mdParse throws. Protects any
  // embedded `<a href="...">` tag from being escaped into visible literal text — see the
  // first bug this file had, where final_text for legal/legal_uk already contains real
  // anchor HTML (legal_citations.format_legal_citation_links runs server-side) mixed into
  // otherwise-plain markdown source.
  var ANY_ANCHOR_RE = /(<a href="([^"]*)"[^>]*>)([\s\S]*?)(<\/a>)/g;

  function _escapeKeepingAnchors(text) {
    var out = "";
    var lastIndex = 0;
    var m;
    ANY_ANCHOR_RE.lastIndex = 0;
    while ((m = ANY_ANCHOR_RE.exec(text))) {
      out += escHtml(text.slice(lastIndex, m.index));
      out += m[0];
      lastIndex = m.index + m[0].length;
    }
    out += escHtml(text.slice(lastIndex));
    return out;
  }

  function _hasAncestorTag(node, tagName) {
    var el = node.parentNode;
    while (el && el.nodeType === 1) {
      if (el.tagName === tagName) return true;
      el = el.parentNode;
    }
    return false;
  }

  var BLOCK_SELECTOR = "p, li, h1, h2, h3, h4, h5, h6, blockquote, td, th, dd, dt";

  /** Converts a flat character offset within blockEl's rendered text into a (node, offset)
   * position usable by Range.setStart/setEnd, by walking the block's text nodes and
   * accumulating length. */
  function _offsetToNodePos(blockEl, offset, isStart) {
    var walker = document.createTreeWalker(blockEl, NodeFilter.SHOW_TEXT, null, false);
    var node;
    var acc = 0;
    var last = null;
    while ((node = walker.nextNode())) {
      var len = node.nodeValue.length;
      // A START offset on a node boundary belongs to the NEXT node (its offset 0), not the
      // end of the previous one -- otherwise a range starting right after a whitespace node
      // (e.g. the "\n" between <blockquote> and its <p>) begins outside the <p>, and
      // extractContents() lifts the whole paragraph into an inline <mark> (found live).
      if (isStart ? offset < acc + len : offset <= acc + len) return { node: node, offset: offset - acc };
      acc += len;
      last = node;
    }
    return last ? { node: last, offset: last.nodeValue.length } : null;
  }

  /** Finds `quote` inside one BLOCK element's flattened text content (a <p>, <li>, <h2>, ...),
   * and wraps exactly that range using a real DOM Range — not a single-text-node substring
   * search. A quote spanning a bold/italic/code boundary lives in MULTIPLE sibling text nodes
   * in the real DOM (the browser fragments text at every inline-element boundary, regardless
   * of what the "visible reading text" looks like), so a single-node search silently missed
   * any sentence containing emphasis. Range.extractContents() + insertNode() handles a range
   * that partially overlaps an element correctly (it splits the element at the boundary),
   * which is exactly the behavior needed here and is far simpler and more correct than
   * manually re-implementing that splitting logic.
   *
   * A quote that doesn't appear in this block at all — markdown rewrote it, the model's
   * wording drifted from what the server extracted — is silently skipped here (the caller
   * tries the next block), never guessed at or force-matched; same fail-closed policy the
   * app's own decision-anchor matching uses. */
  function _wrapQuoteInBlock(blockEl, quote, wrapFn) {
    var text = blockEl.textContent;
    var idx = text.indexOf(quote);
    if (idx === -1) return false;
    var startPos = _offsetToNodePos(blockEl, idx, true);
    var endPos = _offsetToNodePos(blockEl, idx + quote.length, false);
    if (!startPos || !endPos) return false;
    if (_hasAncestorTag(startPos.node, "MARK") || _hasAncestorTag(endPos.node, "MARK")) {
      return false; // already covered by an earlier (longer, higher-priority) span
    }
    var range = document.createRange();
    range.setStart(startPos.node, startPos.offset);
    range.setEnd(endPos.node, endPos.offset);
    var contents = range.extractContents();
    var mark = wrapFn();
    mark.appendChild(contents);
    range.insertNode(mark);
    return true;
  }

  /** Tries every block-level element in document order until `quote` is found in one — a span
   * almost always falls entirely within a single <p>/<li>/<h2>, since confidence.py's
   * sentence extraction splits on the same line/sentence boundaries markdown blocks do. */
  /** Block elements with LEAF blocks first (a <p> inside a <blockquote>/<li> before its
   * container), each in document order. Trying the container first wrapped the inner <p>
   * whole inside an inline <mark> (found live on a quoted judgment); the container is now
   * only a fallback for a quote that genuinely spans several leaf blocks. */
  function _blocksLeafFirst(containerEl, withText) {
    var all = Array.prototype.slice.call(containerEl.querySelectorAll(BLOCK_SELECTOR));
    var leaves = [], parents = [];
    for (var i = 0; i < all.length; i++) {
      (all[i].querySelector(BLOCK_SELECTOR) ? parents : leaves).push(all[i]);
    }
    return leaves.concat(parents).map(function (el) {
      return { el: el, text: withText ? el.textContent : null };
    });
  }

  function _findAndWrapInAnyBlock(containerEl, quote, wrapFn, blockCache) {
    if (!quote) return false;
    // blockCache (optional): [{el, text}] built once per render. Wrapping a range in <mark>
    // never changes a block's textContent, so the cached text stays valid across spans --
    // and skipping blocks that can't contain the quote avoids re-reading textContent
    // spans x blocks times (~160K reads on a long answer once every sentence is scored).
    var blocks = blockCache || _blocksLeafFirst(containerEl, false);
    for (var i = 0; i < blocks.length; i++) {
      if (blocks[i].text !== null && blocks[i].text.indexOf(quote) === -1) continue;
      if (_wrapQuoteInBlock(blocks[i].el, quote, wrapFn)) return true;
    }
    return false;
  }

  var ConfidenceRender = {
    on: false,
    payload: null,
    liveWords: [],

    /** Call at the start of every new assistant turn, before any chunk of that turn arrives. */
    resetTurn: function () {
      this.payload = null;
      this.liveWords = [];
    },

    /** One [CONF]{...}[/CONF] per-token frame. Returns true if it consumed the chunk (the
     * caller's handleChunk must `return` immediately after, exactly like its existing
     * [TRACE] branch — these frames must never reach the text-accumulation fallthrough). */
    handleLiveFrame: function (chunk) {
      if (chunk.indexOf("[CONF]") !== 0) return false;
      var closeAt = chunk.lastIndexOf("[/CONF]");
      if (closeAt === -1) return true; // malformed — still consume it, never render it as text
      try {
        var body = JSON.parse(chunk.slice(6, closeAt));
        this.liveWords.push(body);
      } catch (e) {
        // Tolerate one bad frame; telemetry must never break the turn.
      }
      return true;
    },

    /** The post-__END__ envelope, already JSON-parsed by the caller's own catch-all guard.
     * Returns true if this was a confidence envelope (so the caller's catch-all knows not to
     * fall through to rendering it as text either). */
    handleEnvelope: function (obj) {
      if (!obj || obj.type !== "confidence") return false;
      this.payload = obj;
      return true;
    },

    /** Builds (or rebuilds) the toggle button. onToggle(isOn) is called every time it flips,
     * so the caller can re-render the current bubble immediately. */
    renderToggle: function (container, onToggle) {
      var self = this;
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "confidence-toggle";
      btn.textContent = "Thermo view";
      btn.title = "Toggle confidence shading for this answer";
      btn.addEventListener("click", function () {
        self.on = !self.on;
        btn.classList.toggle("active", self.on);
        onToggle(self.on);
      });
      container.appendChild(btn);
      return btn;
    },

    renderLegend: function (container) {
      if (!container) return;
      if (!this.payload || this.payload.coverage === "none") {
        container.innerHTML =
          '<div class="confidence-legend confidence-unavailable">Confidence shading unavailable for this answer — nothing was checked.</div>';
        return;
      }
      var chips = Object.keys(TIER_STYLE)
        .map(function (tier) {
          var s = TIER_STYLE[tier];
          return (
            '<span class="legend-chip" style="background:' +
            s.bg +
            ";color:" +
            s.fg +
            '">' +
            escHtml(s.label) +
            "</span>"
          );
        })
        .join("");
      container.innerHTML =
        '<div class="confidence-legend"><strong>Confidence shading</strong> shows where each sentence came from, not whether it is correct. ' +
        chips +
        "</div>";
    },

    /** Styles spans directly onto the rendered DOM in `bubbleEl` (already populated by
     * render() via _renderPlainOrMarkdown before this runs). Two passes:
     *   1. Spans from a citation source carry `ref` = the link's href (confidence.py sets
     *      this for every citation span, good or bad). Matched against real <a href> elements
     *      already in the DOM — marked.parse() renders BOTH chat-wonder's pre-embedded
     *      `<a class="legal-ref...">` tags AND any markdown-native `[label](url)` link the
     *      same way, as real anchor elements, so one href-match covers both forms. Styles the
     *      anchor in place: border/background only, href/target/class never touched.
     *   2. Everything else matches by quote TEXT against text nodes (never inside an <a>,
     *      which avoids ever splitting an anchor — the root cause of the original escaping
     *      bug). Longest quote first, so a longer, more specific match is not pre-empted by a
     *      short substring of it. */
    _applySpansToDom: function (bubbleEl, spans) {
      var remaining = [];
      for (var i = 0; i < spans.length; i++) {
        var span = spans[i];
        var wrapped = false;
        if (span.ref) {
          var anchors = bubbleEl.querySelectorAll('a[href="' + _cssEscape(span.ref) + '"]');
          if (anchors.length) {
            var style = TIER_STYLE[span.tier] || TIER_STYLE.UNSOURCED;
            var title = (style.label || "") + (span.why ? " — " + span.why : "");
            for (var a = 0; a < anchors.length; a++) {
              anchors[a].style.background = style.bg;
              anchors[a].style.color = style.fg;
              anchors[a].style.borderRadius = "3px";
              anchors[a].style.padding = "0 2px";
              anchors[a].title = title;
            }
            wrapped = true;
          }
        }
        if (!wrapped) remaining.push(span);
      }
      remaining.sort(function (x, y) {
        return (y.quote || "").length - (x.quote || "").length;
      });
      var blockCache = _blocksLeafFirst(bubbleEl, true);
      for (var j = 0; j < remaining.length; j++) {
        (function (span) {
          _findAndWrapInAnyBlock(bubbleEl, span.quote, function () {
            var style = TIER_STYLE[span.tier] || TIER_STYLE.UNSOURCED;
            var title = (style.label || "") + (span.why ? " — " + span.why : "");
            var mark = document.createElement("mark");
            mark.className = "conf-span";
            mark.style.background = style.bg;
            mark.style.color = style.fg;
            mark.title = title;
            return mark;
          }, blockCache);
        })(remaining[j]);
      }
    },

    /** Per-word entropy coloring, applied the same DOM-walking way: collects every text node
     * first (mutating mid-walk breaks a live TreeWalker), then re-runs the existing
     * string-level word colorer on each node's own text and splices the result back in,
     * preserving whatever markdown structure (headings, bold, etc.) that node already sits
     * inside. `words` is already word-grouped and MAX-aggregated server-side
     * (confidence.group_words). */
    _applyWordEntropyToDom: function (bubbleEl, words) {
      var walker = document.createTreeWalker(bubbleEl, NodeFilter.SHOW_TEXT, null, false);
      var nodes = [];
      var node;
      while ((node = walker.nextNode())) {
        if (_hasAncestorTag(node, "A")) continue;
        nodes.push(node);
      }
      for (var i = 0; i < nodes.length; i++) {
        var n = nodes[i];
        var plainEscaped = escHtml(n.nodeValue);
        var html = this._wordEntropyHtml(n.nodeValue, words);
        if (html === plainEscaped) continue; // nothing in this node matched — leave it alone
        var wrapper = document.createElement("span");
        wrapper.innerHTML = html;
        var frag = document.createDocumentFragment();
        while (wrapper.firstChild) frag.appendChild(wrapper.firstChild);
        n.parentNode.replaceChild(frag, n);
      }
    },

    _wordEntropyHtml: function (rawText, words) {
      var out = "";
      var cursor = 0;
      var LN5 = Math.log(5);
      for (var i = 0; i < words.length; i++) {
        var w = words[i];
        var idx = rawText.indexOf(w.w, cursor);
        if (idx === -1) continue;
        out += escHtml(rawText.slice(cursor, idx));
        var color = entropyColor(w.hn);
        var purple = w.h != null && w.h >= LN5 * ENTROPY_PURPLE_HN;
        var styleParts = [];
        if (color) styleParts.push("color:" + color);
        if (purple) styleParts.push("background:rgba(168,85,247,0.22)");
        var wordHtml = escHtml(w.w);
        if (styleParts.length) {
          out +=
            '<span class="conf-word" style="' +
            styleParts.join(";") +
            '" title="h=' +
            w.h +
            " nats, hn=" +
            w.hn +
            '">' +
            wordHtml +
            "</span>";
        } else {
          out += wordHtml;
        }
        cursor = idx + w.w.length;
      }
      out += escHtml(rawText.slice(cursor));
      return out;
    },

    /** Live per-token frames carry one token each (confidence.to_live_frame), not pre-grouped
     * words — this does the same leading-whitespace grouping + MAX aggregation that
     * confidence.group_words does server-side, so toggling mid-stream still colors correctly
     * even before the final envelope's tokens.words arrives. */
    _liveWordsAsPayloadWords: function () {
      var words = [];
      var curText = "";
      var curHs = [];
      function flush() {
        var text = curText.trim();
        if (!text) return;
        var measured = curHs.filter(function (h) {
          return h != null;
        });
        var h = measured.length ? Math.max.apply(null, measured) : null;
        words.push({ w: text, h: h, hn: h != null ? h / Math.log(5) : null });
      }
      for (var i = 0; i < this.liveWords.length; i++) {
        var tok = this.liveWords[i];
        var t = tok.token || "";
        var isNew = t.length > 0 && /^[\s]/.test(t) && curText;
        if (isNew) {
          flush();
          curText = "";
          curHs = [];
        }
        curText += t;
        curHs.push(tok.h);
      }
      flush();
      return words;
    },

    _renderPlainOrMarkdown: function (bubbleEl, rawText, mdParse) {
      if (typeof mdParse === "function") {
        try {
          bubbleEl.innerHTML = mdParse(rawText);
          return;
        } catch (e) {
          // fall through to the escaped-but-anchor-safe path below
        }
      }
      bubbleEl.innerHTML = _escapeKeepingAnchors(rawText);
    },

    /** Renders `rawText` into `bubbleEl`. Markdown (or escaped-but-anchor-safe plain text, see
     * _renderPlainOrMarkdown) is ALWAYS rendered first, unconditionally — toggle off, toggle
     * on with no spans, toggle on with spans, all start from the same rendered structure. When
     * on, spans/words are then overlaid onto that real DOM rather than the pre-render string.
     * See this file's top-of-file comment for why that ordering is the fix, not a detail. */
    render: function (bubbleEl, rawText, legendContainer, mdParse) {
      this._renderPlainOrMarkdown(bubbleEl, rawText, mdParse);
      if (!this.on) {
        if (legendContainer) legendContainer.innerHTML = "";
        return;
      }
      this.renderLegend(legendContainer);
      var spans = this.payload && this.payload.spans;
      var words =
        (this.payload && this.payload.tokens && this.payload.tokens.words) ||
        (this.liveWords.length ? this._liveWordsAsPayloadWords() : null);
      if (spans && spans.length) {
        this._applySpansToDom(bubbleEl, spans);
      } else if (words && words.length) {
        this._applyWordEntropyToDom(bubbleEl, words);
      }
    },
  };

  function _cssEscape(s) {
    // Minimal: hrefs here are always http(s) URLs: escape only the one character
    // (the double quote) that could break the attribute-selector string.
    return String(s).replace(/"/g, '\\"');
  }

  window.ConfidenceRender = ConfidenceRender;
})();
