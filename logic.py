import random
import re
import string
import graphviz

try:
    import regex
    REGEX_AVAILABLE = True
except ImportError:
    REGEX_AVAILABLE = False
    print("Warning: 'regex' module not available. Install with: pip install regex")
    print("Falling back to standard 're' module (no recursion support)\n")

# --- Configuration ---
MAX_LOOP_UNROLL = 3  # How many times to unroll *, + (e.g., 0, 1, 2, 3)
MAX_RECURSION_DEPTH = 0  # Maximum depth for recursive patterns (?R)

class Node:
    def __init__(self, label, kind='simple', payload=None):
        self.id = None  # Assigned later
        self.label = label
        self.kind = kind
        self.payload = payload
        self.next = []
        self.char_count = 1

    def connect(self, node):
        self.next.append(node)
        return node

    def __repr__(self):
        return f"[{self.label}]"

class RegexGraphBuilder:
    def __init__(self, regex_pattern, flags=0):
        self.regex = regex_pattern
        self.flags = flags
        self.start_node = Node("START", kind='start')
        self.end_node = Node("END", kind='end')
        self.nodes = []
        self.recursion_counter = 0
        self.valid_samples_cache = None  # Cache for mutation-based negatives
        self.conditionals = {}  # Store conditional info: {marker: (group_id, yes, no)}
        self.lookarounds = []  # Store lookaround assertions
        
        # Try to use regex module first for advanced features
        if REGEX_AVAILABLE:
            self.using_regex_module = True
            self._build_with_regex_module()
        else:
            self.using_regex_module = False
            self._build_with_re_module()
        
        self._assign_ids()

    def _build_with_regex_module(self):
        """Build using the advanced regex module"""
        try:
            import regex as re_module
            # Parse the pattern
            parsed = re_module.compile(self.regex, self.flags)
            # Get the parsed structure
            pattern_obj = parsed.pattern
            
            # First expand Unicode properties
            expanded = self._expand_unicode_properties(self.regex)
            
            # Detect lookarounds (store for negative generation, but don't expand)
            expanded, has_lookarounds = self._detect_and_expand_lookarounds(expanded)
            
            # Detect and handle conditionals - store them for later processing
            expanded, has_conditionals = self._detect_and_expand_conditionals(expanded)
            
            # Then expand recursion
            expanded = self._expand_recursion(expanded, MAX_RECURSION_DEPTH)
            
            # Now parse with standard library since recursion is expanded
            import sre_parse
            parsed = sre_parse.parse(expanded, self.flags)
            data = parsed.data if hasattr(parsed, 'data') else parsed
            
            last = self._build(data, self.start_node)
            for l in last:
                l.connect(self.end_node)
                
        except Exception as e:
            print(f"Parse Error with regex module: {e}")
            print("Falling back to standard re module...")
            self.using_regex_module = False
            self._build_with_re_module()

    def _expand_recursion(self, pattern, max_depth):
        """
        Manually expand (?R) recursion to a fixed depth.
        This is a simplified expansion - proper implementation would need full parser.
        """
        if '(?R)' not in pattern and '(?0)' not in pattern:
            return pattern
        
        # For the simple case of r'\((?:[^()]|(?R))*\)'
        # We'll expand it iteratively
        
        # Base case (depth 0): match empty or non-parens
        base = r'\([^()]*\)'
        
        # Build depth by depth
        current = base
        for depth in range(1, max_depth + 1):
            # Replace (?R) with the previous level pattern
            # For \((?:[^()]|(?R))*\), we replace (?R) with previous expansion
            next_pattern = pattern.replace('(?R)', current).replace('(?0)', current)
            current = next_pattern
        
        return current

    def _expand_unicode_properties(self, pattern):
        """
        Convert Unicode property escapes like \p{L} to character classes.
        The regex module supports these, but sre_parse doesn't.
        """
        import re as re_std
        
        # Common Unicode property mappings (simplified)
        unicode_props = {
            r'\p{L}': r'[a-zA-Z\u00C0-\u024F\u1E00-\u1EFF]',  # Letters (Latin extended)
            r'\p{Ll}': r'[a-z\u00E0-\u00FF]',  # Lowercase letters
            r'\p{Lu}': r'[A-Z\u00C0-\u00DE]',  # Uppercase letters
            r'\p{N}': r'[0-9]',  # Numbers (simplified to ASCII digits)
            r'\p{Nd}': r'[0-9]',  # Decimal numbers
            r'\p{P}': r'[!-/:-@\[-`{-~]',  # Punctuation
            r'\p{S}': r'[$+<->\^`|~]',  # Symbols
            r'\p{Z}': r'[ \t\n\r]',  # Separators (whitespace)
            r'\p{Zs}': r'[ ]',  # Space separator
        }
        
        # Also handle negated properties \P{...}
        negated_props = {
            r'\P{L}': r'[^a-zA-Z\u00C0-\u024F\u1E00-\u1EFF]',
            r'\P{N}': r'[^0-9]',
            r'\P{Nd}': r'[^0-9]',
        }
        
        result = pattern
        
        # Replace Unicode properties
        for prop, replacement in unicode_props.items():
            result = result.replace(prop, replacement)
        
        for prop, replacement in negated_props.items():
            result = result.replace(prop, replacement)
        
        return result
    
    def _detect_and_expand_conditionals(self, pattern):
        """
        Detect and expand conditional patterns like (?(1)yes|no).
        Since sre_parse doesn't support these, we need special handling.
        We'll expand them into equivalent alternations based on captured groups.
        """
        import re as re_std
        
        # Match patterns like (?(1)yes|no) or (?(1)yes)
        conditional_pattern = r'\(\?\((\d+)\)([^|)]+)(?:\|([^)]+))?\)'
        
        matches = list(re_std.finditer(conditional_pattern, pattern))
        
        if not matches:
            return pattern, False
        
        # For now, we'll use a heuristic approach:
        # (a)?(?(1)b|c) becomes:
        # - Path 1: ab (when group 1 matches)
        # - Path 2: c (when group 1 doesn't match)
        
        # Since we can't properly model conditionals in the graph without runtime state,
        # we'll expand to all valid possibilities
        
        # Store conditional info for generation
        for match in matches:
            group_id = int(match.group(1))
            yes_branch = match.group(2)
            no_branch = match.group(3) if match.group(3) else ''
            
            marker = f'__COND_{group_id}_{yes_branch}_{no_branch}__'
            self.conditionals[marker] = (group_id, yes_branch, no_branch)
        
        # For graph building, we need to create both paths
        # Replace (?(1)b|c) with (?:b|c) but mark it specially
        def replace_conditional(match):
            group_id = match.group(1)
            yes_branch = match.group(2)
            no_branch = match.group(3) if match.group(3) else ''
            
            # Create a non-capturing group with both branches
            # We'll add a special marker node later
            if no_branch:
                return f'(?:_COND{group_id}Y_{yes_branch}|_COND{group_id}N_{no_branch})'
            else:
                # If no "no" branch, it's optional
                return f'(?:_COND{group_id}Y_{yes_branch})?'
        
        result = re_std.sub(conditional_pattern, replace_conditional, pattern)
        return result, True
    
    def _detect_and_expand_lookarounds(self, pattern):
        """
        Detect lookahead/lookbehind assertions and mark them for special handling.
        Lookarounds are zero-width assertions that don't consume characters.
        
        Supported:
        (?=...)  - Positive lookahead
        (?!...)  - Negative lookahead  
        (?<=...) - Positive lookbehind
        (?<!...) - Negative lookbehind
        """
        import re as re_std
        
        # Detect all lookaround types
        lookahead_pos = r'\(\?=([^)]+)\)'      # (?=...)
        lookahead_neg = r'\(\?!([^)]+)\)'      # (?!...)
        lookbehind_pos = r'\(\?<=([^)]+)\)'    # (?<=...)
        lookbehind_neg = r'\(\?<!([^)]+)\)'    # (?<!...)
        
        has_lookarounds = False
        
        # Check for each type and store info
        for pattern_type, regex_pattern, assertion_type in [
            ('positive_lookahead', lookahead_pos, 'must_follow'),
            ('negative_lookahead', lookahead_neg, 'must_not_follow'),
            ('positive_lookbehind', lookbehind_pos, 'must_precede'),
            ('negative_lookbehind', lookbehind_neg, 'must_not_precede'),
        ]:
            matches = list(re_std.finditer(regex_pattern, pattern))
            if matches:
                has_lookarounds = True
                for match in matches:
                    content = match.group(1)
                    marker = f'__{pattern_type.upper()}_{content}__'
                    if not hasattr(self, 'lookarounds'):
                        self.lookarounds = []
                    self.lookarounds.append({
                        'type': pattern_type,
                        'assertion': assertion_type,
                        'content': content,
                        'marker': marker
                    })
        
        # For now, we'll mark lookarounds but keep them in pattern
        # The regex module will handle them, we'll use them for negative generation
        return pattern, has_lookarounds

    def _build_with_re_module(self):
        """Build using standard re module (no recursion support)"""
        import sre_parse
        try:
            # First expand Unicode properties (standard re doesn't support them either)
            expanded = self._expand_unicode_properties(self.regex)
            
            # Detect lookarounds
            expanded, has_lookarounds = self._detect_and_expand_lookarounds(expanded)
            
            # Detect and handle conditionals
            expanded, has_conditionals = self._detect_and_expand_conditionals(expanded)
            
            parsed = sre_parse.parse(expanded, self.flags)
            data = parsed.data if hasattr(parsed, 'data') else parsed
            
            last = self._build(data, self.start_node)
            for l in last:
                l.connect(self.end_node)
        except Exception as e:
            print(f"Parse Error: {e}")

    
    def _assign_ids(self):
        """Assign unique IDs for visualization with safety limit"""
        visited = set()
        stack = [self.start_node]
        counter = 0
        while stack:
            node = stack.pop()
            if node in visited: continue
            visited.add(node)
            
            # SAFETY LIMIT: Prevent graph explosion
            if counter > 500:
                raise Exception(
                    "Regex is too complex to visualize safely. "
                    "Please check our CLI tool for complex regex: "
                    "https://github.com/your-username/your-cli-tool"
                )
                
            node.id = str(counter)
            counter += 1
            stack.extend(node.next)
            self.nodes.append(node)

    # def _assign_ids(self):
    #     """Assign unique IDs for visualization"""
    #     visited = set()
    #     stack = [self.start_node]
    #     counter = 0
    #     while stack:
    #         node = stack.pop()
    #         if node in visited: continue
    #         visited.add(node)
    #         node.id = str(counter)
    #         counter += 1
    #         stack.extend(node.next)
    #         self.nodes.append(node)

    def _get_category_chars(self, category_code):
        """Convert category codes to character sets"""
        import sre_constants
        if category_code == sre_constants.CATEGORY_DIGIT:
            return set(string.digits), 10, "\\d"
        elif category_code == sre_constants.CATEGORY_NOT_DIGIT:
            all_printable = set(string.printable)
            return all_printable - set(string.digits), 90, "\\D"
        elif category_code == sre_constants.CATEGORY_SPACE:
            return set(string.whitespace), 6, "\\s"
        elif category_code == sre_constants.CATEGORY_NOT_SPACE:
            all_printable = set(string.printable)
            return all_printable - set(string.whitespace), 94, "\\S"
        elif category_code == sre_constants.CATEGORY_WORD:
            word_chars = set(string.ascii_letters + string.digits + '_')
            return word_chars, 63, "\\w"
        elif category_code == sre_constants.CATEGORY_NOT_WORD:
            all_printable = set(string.printable)
            word_chars = set(string.ascii_letters + string.digits + '_')
            return all_printable - word_chars, 37, "\\W"
        else:
            return set(), 0, f"CAT_{category_code}"

    def _build(self, tree, start_node):
        """Build graph from parse tree"""
        import sre_constants
        
        current_leads = [start_node]
        items = tree if isinstance(tree, (list, tuple)) else tree.data
        
        for op, av in items:
            new_leads = []
            
            for lead in current_leads:
                
                # --- LITERAL (Char) ---
                if op == sre_constants.LITERAL:
                    n = Node(f"'{chr(av)}'", kind='match', payload=chr(av))
                    n.char_count = 1
                    lead.connect(n)
                    new_leads.append(n)

                # --- ANY (.) ---
                elif op == sre_constants.ANY:
                    char_set = set(string.printable) - {'\n', '\r'}
                    n = Node(".", kind='class', payload=char_set)
                    n.char_count = len(char_set)
                    lead.connect(n)
                    new_leads.append(n)

                # --- CATEGORY (\d, \w, \s, etc) ---
                elif op == sre_constants.CATEGORY:
                    char_set, count, label = self._get_category_chars(av)
                    n = Node(label, kind='class', payload=char_set)
                    n.char_count = count
                    lead.connect(n)
                    new_leads.append(n)

                # --- OR (Branch) ---
                elif op == sre_constants.BRANCH:
                    split = Node("Split", kind='split')
                    lead.connect(split)
                    
                    for alt_path in av[1]:
                        ends = self._build(alt_path, split)
                        new_leads.extend(ends)

                # --- CAPTURE GROUP ---
                elif op == sre_constants.SUBPATTERN:
                    group_num = av[0]
                    sub_tree = av[3]
                    
                    if group_num is None:
                        # Non-capturing group (?:...)
                        inner_ends = self._build(sub_tree, lead)
                        new_leads.extend(inner_ends)
                    else:
                        # Capturing group
                        cap_start = Node(f"Start G{group_num}", kind='group_start', payload=group_num)
                        lead.connect(cap_start)
                        
                        inner_ends = self._build(sub_tree, cap_start)
                        
                        for end in inner_ends:
                            cap_end = Node(f"End G{group_num}", kind='group_end', payload=group_num)
                            end.connect(cap_end)
                            new_leads.append(cap_end)

                # --- MAX_REPEAT (*, +, {n,m}) - GREEDY ---
                elif op == sre_constants.MAX_REPEAT:
                    min_rep, max_rep = av[0], av[1]
                    sub_tree = av[2]
                    
                    if max_rep == sre_constants.MAXREPEAT:
                        max_rep = MAX_LOOP_UNROLL

                    split = Node(f"Loop {min_rep}-{max_rep}", kind='split')
                    lead.connect(split)

                    for i in range(min_rep, max_rep + 1):
                        if i == 0:
                            new_leads.append(split) 
                        else:
                            chain_leads = [split]
                            for _ in range(i):
                                next_leads = []
                                for l_node in chain_leads:
                                    ends = self._build(sub_tree, l_node)
                                    next_leads.extend(ends)
                                chain_leads = next_leads
                            new_leads.extend(chain_leads)

                # --- MIN_REPEAT (non-greedy *?, +?, {n,m}?) ---
                elif op == sre_constants.MIN_REPEAT:
                    min_rep, max_rep = av[0], av[1]
                    sub_tree = av[2]
                    
                    if max_rep == sre_constants.MAXREPEAT:
                        max_rep = MAX_LOOP_UNROLL

                    split = Node(f"Loop? {min_rep}-{max_rep}", kind='split')
                    lead.connect(split)

                    for i in range(min_rep, max_rep + 1):
                        if i == 0:
                            new_leads.append(split) 
                        else:
                            chain_leads = [split]
                            for _ in range(i):
                                next_leads = []
                                for l_node in chain_leads:
                                    ends = self._build(sub_tree, l_node)
                                    next_leads.extend(ends)
                                chain_leads = next_leads
                            new_leads.extend(chain_leads)

                # --- BACKREFERENCE ---
                elif op == sre_constants.GROUPREF:
                    ref_id = av
                    n = Node(f"\\{ref_id}", kind='backref', payload=ref_id)
                    lead.connect(n)
                    new_leads.append(n)
                
                # --- CHAR CLASS [abc] or [a-z] ---
                elif op == sre_constants.IN:
                    char_set = set()
                    negated = False
                    
                    for sub_op, sub_av in av:
                        if sub_op == sre_constants.LITERAL:
                            char_set.add(chr(sub_av))
                        elif sub_op == sre_constants.RANGE:
                            start, end = sub_av
                            char_set.update(chr(c) for c in range(start, end + 1))
                        elif sub_op == sre_constants.CATEGORY:
                            cat_chars, _, _ = self._get_category_chars(sub_av)
                            char_set.update(cat_chars)
                        elif sub_op == sre_constants.NEGATE:
                            negated = True
                    
                    if negated:
                        all_chars = set(string.printable)
                        char_set = all_chars - char_set
                        label = f"[^...] ({len(char_set)})"
                    else:
                        label = f"[...] ({len(char_set)})"
                    
                    n = Node(label, kind='class', payload=char_set)
                    n.char_count = len(char_set)
                    lead.connect(n)
                    new_leads.append(n)

                # --- ANCHORS ---
                elif op == sre_constants.AT:
                    anchor_type = av
                    if anchor_type == sre_constants.AT_BEGINNING:
                        n = Node("^", kind='anchor', payload='start')
                    elif anchor_type == sre_constants.AT_END:
                        n = Node("$", kind='anchor', payload='end')
                    elif anchor_type == sre_constants.AT_BOUNDARY:
                        n = Node("\\b", kind='anchor', payload='word_boundary')
                    elif anchor_type == sre_constants.AT_BOUNDARY_NOT:
                        n = Node("\\B", kind='anchor', payload='not_word_boundary')
                    else:
                        n = Node(f"Anchor({anchor_type})", kind='anchor', payload=anchor_type)
                    
                    n.char_count = 0
                    lead.connect(n)
                    new_leads.append(n)

                # --- NOT_LITERAL ---
                elif op == sre_constants.NOT_LITERAL:
                    excluded_char = chr(av)
                    all_chars = set(string.printable)
                    char_set = all_chars - {excluded_char}
                    n = Node(f"[^{excluded_char}]", kind='class', payload=char_set)
                    n.char_count = len(char_set)
                    lead.connect(n)
                    new_leads.append(n)

                # --- Unknown ---
                else:
                    n = Node(f"?({op})", kind='unknown', payload=av)
                    lead.connect(n)
                    new_leads.append(n)
            
            current_leads = new_leads

        return current_leads

    def count_paths(self):
        """Count total number of valid strings"""
        memo = {}
        
        def count_from_node(node, has_backref=False):
            if node.kind == 'end':
                return 1
            
            if not has_backref and node in memo:
                return memo[node]
            
            total = 0
            
            if node.kind == 'backref':
                has_backref = True
                for next_node in node.next:
                    total += count_from_node(next_node, has_backref)
            
            elif node.kind in ['match', 'class', 'any']:
                for next_node in node.next:
                    total += node.char_count * count_from_node(next_node, has_backref)
            
            elif node.kind in ['split', 'group_start', 'group_end', 'start', 'anchor']:
                for next_node in node.next:
                    total += count_from_node(next_node, has_backref)
            
            else:
                for next_node in node.next:
                    total += count_from_node(next_node, has_backref)
            
            if not has_backref:
                memo[node] = total
            
            return total
        
        total_count = count_from_node(self.start_node)
        has_backref = any(n.kind == 'backref' for n in self.nodes)
        
        return total_count, has_backref

    def visualize(self, filename='regex_graph', view_on_desktop=False):
        dot = graphviz.Digraph(format='png')
        dot.attr(rankdir='LR')
        
        for n in self.nodes:
            color = 'black'
            shape = 'box'
            style = ''
            fillcolor = 'white'
            label = str(n.label)
            
            if n.kind in ['class', 'any'] and n.char_count > 1:
                label += f"\n({n.char_count} chars)"
            
            if n.kind == 'match': 
                color = 'green'
            elif n.kind == 'class' or n.kind == 'any':
                color = 'purple'
                style = 'filled'
                fillcolor = '#f3e5f5'
            elif n.kind == 'group_start': 
                shape, color = 'component', 'blue'
                style = 'filled'
                fillcolor = '#e3f2fd'
            elif n.kind == 'group_end': 
                shape, color = 'component', 'blue'
                style = 'filled'
                fillcolor = '#e3f2fd'
            elif n.kind == 'backref': 
                shape, style, color = 'note', 'filled', 'orange'
                fillcolor = '#fff3e0'
            elif n.kind == 'split': 
                shape = 'diamond'
            elif n.kind == 'anchor':
                shape = 'hexagon'
                style = 'filled'
                fillcolor = '#e8f5e9'
            elif n.kind == 'start': 
                shape = 'doublecircle'
                style = 'filled'
                fillcolor = '#eeeeee'
            elif n.kind == 'end': 
                shape = 'doublecircle'
                style = 'filled'
                fillcolor = '#eeeeee'

            dot.node(n.id, label, shape=shape, color=color, style=style, fillcolor=fillcolor)

            for nxt in n.next:
                dot.edge(n.id, nxt.id)
        
        if view_on_desktop:
            dot.render(filename, view=True)
            
        return dot

    def generate_sample(self, max_samples=None, force_complete=False):
        """
        Generate sample strings with intelligent sampling strategy.
        
        Strategy:
        1. Count total possible strings
        2. If count <= 1M (or max_samples if provided): Generate ALL strings exhaustively
        3. If count > 1M: Use stratified random sampling for diversity
        
        Args:
            max_samples: Maximum number of samples to generate (default: 1M)
            force_complete: If True, generate all strings even if > max_samples (use with caution)
        
        Returns:
            List of generated strings (sorted)
        """
        if max_samples is None:
            max_samples = 1_000_000
        
        # Count total possible strings
        total_count, has_backref = self.count_paths()
        
        # Decision: exhaustive vs sampling
        if total_count <= max_samples and not has_backref:
            # EXHAUSTIVE GENERATION - generate all strings
            print(f"Generating all {total_count} strings exhaustively...")
            return self._generate_exhaustive()
        else:
            # SAMPLING - use stratified random sampling for diversity
            target = min(max_samples, total_count) if not has_backref else max_samples
            print(f"Total possible: {total_count}, sampling {target} diverse strings...")
            return self._generate_stratified_sample(target)
    
    def _generate_exhaustive(self):
        """
        Generate ALL possible strings by exhaustive graph traversal.
        Optimized for speed with iterative approach and set deduplication.
        """
        results = set()
        
        # Stack: (node, string, memory, active_captures)
        stack = [(self.start_node, "", {}, {})]
        
        while stack:
            node, s, memory, active_caps = stack.pop()
            
            if node.kind == 'end':
                # Clean up conditional markers
                clean_s = self._clean_conditional_markers(s, memory)
                if clean_s is not None:
                    results.add(clean_s)
                continue
            
            # Process node and generate next states
            next_states = self._get_next_states(node, s, memory, active_caps)
            stack.extend(next_states)
        
        return list(results)
    
    def _generate_stratified_sample(self, target_count):
        """
        Generate diverse samples using stratified random sampling.
        
        Strategy:
        1. Identify "choice points" in the graph (splits, classes)
        2. At each choice point, randomly select to ensure diversity
        3. Use weighted sampling based on downstream path counts
        4. Deduplicate and ensure we hit target count
        """
        import random
        
        results = set()
        attempts = 0
        max_attempts = target_count * 5  # Allow some retries for deduplication
        
        # Pre-compute path counts from each node (for weighted sampling)
        path_counts = self._compute_path_counts_from_nodes()
        
        while len(results) < target_count and attempts < max_attempts:
            attempts += 1
            
            # Generate one random path through the graph
            sample = self._generate_random_path(path_counts, random)
            if sample:
                results.add(sample)
        
        # If we still need more samples, fall back to BFS with random selection
        if len(results) < target_count:
            results.update(self._generate_bfs_sample(target_count - len(results)))
        
        return sorted(list(results))[:target_count]
    
    def _get_next_states(self, node, s, memory, active_caps):
        """
        Get all possible next states from current node.
        Returns list of (next_node, next_string, next_memory, next_active_caps)
        """
        next_states = []
        
        if node.kind == 'match':
            next_s = s + node.payload
            for nxt in node.next:
                next_states.append((nxt, next_s, memory.copy(), active_caps.copy()))
        
        elif node.kind in ['class', 'any']:
            # For exhaustive generation, explore ALL characters in the set
            char_set = node.payload
            for char in char_set:
                next_s = s + char
                for nxt in node.next:
                    next_states.append((nxt, next_s, memory.copy(), active_caps.copy()))
        
        elif node.kind == 'group_start':
            next_active = active_caps.copy()
            next_active[node.payload] = len(s)
            for nxt in node.next:
                next_states.append((nxt, s, memory.copy(), next_active))
        
        elif node.kind == 'group_end':
            next_mem = memory.copy()
            start_idx = active_caps.get(node.payload)
            if start_idx is not None:
                captured = s[start_idx:]
                next_mem[node.payload] = captured
            for nxt in node.next:
                next_states.append((nxt, s, next_mem, active_caps.copy()))
        
        elif node.kind == 'backref':
            group_id = node.payload
            next_s = s
            if group_id in memory:
                next_s += memory[group_id]
            for nxt in node.next:
                next_states.append((nxt, next_s, memory.copy(), active_caps.copy()))
        
        elif node.kind in ['split', 'anchor', 'start', 'group_start', 'group_end']:
            # Pass through nodes - just continue
            for nxt in node.next:
                next_states.append((nxt, s, memory.copy(), active_caps.copy()))
        
        else:
            # Unknown node type - pass through
            for nxt in node.next:
                next_states.append((nxt, s, memory.copy(), active_caps.copy()))
        
        return next_states
    
    def _compute_path_counts_from_nodes(self):
        """
        Compute the number of paths from each node to END.
        Used for weighted random sampling to ensure diversity.
        """
        memo = {}
        
        def count_from_node(node, visited=None):
            if visited is None:
                visited = set()
            
            if node in visited:  # Cycle detection
                return 1  # Treat cycles as single path
            
            if node.kind == 'end':
                return 1
            
            if node in memo:
                return memo[node]
            
            visited = visited | {node}
            total = 0
            
            if node.kind in ['match', 'class', 'any']:
                char_count = node.char_count if hasattr(node, 'char_count') else 1
                for next_node in node.next:
                    total += char_count * count_from_node(next_node, visited)
            else:
                for next_node in node.next:
                    total += count_from_node(next_node, visited)
            
            memo[node] = max(total, 1)  # Ensure at least 1
            return memo[node]
        
        # Compute for all nodes
        for node in self.nodes:
            count_from_node(node)
        
        return memo
    
    def _generate_random_path(self, path_counts, random):
        """
        Generate one random path through the graph.
        Uses weighted selection at choice points based on downstream path counts.
        """
        node = self.start_node
        s = ""
        memory = {}
        active_caps = {}
        
        max_steps = 1000  # Prevent infinite loops
        steps = 0
        
        while node.kind != 'end' and steps < max_steps:
            steps += 1
            
            if node.kind == 'match':
                s += node.payload
                if node.next:
                    node = node.next[0]
                else:
                    break
            
            elif node.kind in ['class', 'any']:
                # Randomly select a character from the set
                char_set = list(node.payload)
                if char_set:
                    s += random.choice(char_set)
                if node.next:
                    node = node.next[0]
                else:
                    break
            
            elif node.kind == 'group_start':
                active_caps[node.payload] = len(s)
                if node.next:
                    node = node.next[0]
                else:
                    break
            
            elif node.kind == 'group_end':
                start_idx = active_caps.get(node.payload)
                if start_idx is not None:
                    memory[node.payload] = s[start_idx:]
                if node.next:
                    node = node.next[0]
                else:
                    break
            
            elif node.kind == 'backref':
                group_id = node.payload
                if group_id in memory:
                    s += memory[group_id]
                if node.next:
                    node = node.next[0]
                else:
                    break
            
            elif node.kind == 'split':
                # Check if this is a loop node (quantifier)
                if 'Loop' in node.label:
                    # Extract min and max from label like "Loop 1-3"
                    try:
                        parts = node.label.split()
                        if len(parts) > 1:
                            range_part = parts[1].replace('?', '')
                            min_rep = int(range_part.split('-')[0])
                            max_rep = int(range_part.split('-')[1])
                            
                            # Use smart quantifier sampling
                            chosen_count = self._choose_repeat_count(min_rep, max_rep, random)
                            
                            # Find the path that represents the chosen count
                            # In our graph, we have branches for 0, 1, 2, ... max_rep iterations
                            # We need to pick the branch that matches chosen_count
                            
                            # Since we unrolled loops, each branch represents a specific count
                            # Branch 0: skip (0 iterations)
                            # Branch 1: 1 iteration
                            # Branch 2: 2 iterations, etc.
                            
                            if node.next and chosen_count < len(node.next):
                                node = node.next[chosen_count]
                            elif node.next:
                                # If chosen_count exceeds branches, take the last one
                                node = node.next[-1]
                            else:
                                break
                            continue
                    except:
                        pass
                
                # Regular split (not a loop) - weighted random choice
                if node.next:
                    # Weight by downstream path counts
                    weights = [path_counts.get(n, 1) for n in node.next]
                    total_weight = sum(weights)
                    if total_weight > 0:
                        # Normalize and choose
                        probs = [w / total_weight for w in weights]
                        node = random.choices(node.next, weights=probs)[0]
                    else:
                        node = random.choice(node.next)
                else:
                    break
            
            else:
                # Pass-through nodes
                if node.next:
                    node = node.next[0]
                else:
                    break
        
        # Clean up conditional markers
        clean_s = self._clean_conditional_markers(s, memory)
        return clean_s
    
    def _choose_repeat_count(self, min_rep, max_rep, rng):
        """
        Smart quantifier sampling rules for diverse test case generation.
        
        Patterns:
        * or {0,}        -> 0, 1, 2              (common cases)
        + or {1,}        -> 1, 2, 3              (common cases)
        ? or {0,1}       -> 0, 1, 2              (2 is overflow for negative tests)
        {0,n} or {,n}    -> 0, 1, n, n+1         (min, typical, max, overflow)
        {m,n}            -> m, random(m,n), n    (min, middle, max)
        {m} (m>0)        -> 1, random(1,m), m    (underflow, middle, exact)
        {m,} (m>1)       -> m, m+random(0..3)    (min, slightly more)
        
        Args:
            min_rep: Minimum repetitions
            max_rep: Maximum repetitions (could be MAX_LOOP_UNROLL for unbounded)
            rng: Random number generator
            
        Returns:
            Chosen repetition count
        """
        # Detect pattern type based on min/max
        
        # Pattern: * or {0,}
        if min_rep == 0 and max_rep >= MAX_LOOP_UNROLL:
            return rng.choice([0, 1, 2])
        
        # Pattern: + or {1,}
        elif min_rep == 1 and max_rep >= MAX_LOOP_UNROLL:
            return rng.choice([1, 2, 3])
        
        # Pattern: ? or {0,1}
        elif min_rep == 0 and max_rep == 1:
            return rng.choice([0, 1, 2])  # 2 is overflow for testing
        
        # Pattern: {0,n} where n is specific
        elif min_rep == 0 and max_rep < MAX_LOOP_UNROLL:
            choices = [0, 1, max_rep]
            if max_rep > 1:
                choices.append(max_rep + 1)  # overflow case
            return rng.choice(choices)
        
        # Pattern: {m,n} with both bounds
        elif min_rep > 0 and max_rep < MAX_LOOP_UNROLL and max_rep > min_rep:
            # Choose from: min, random middle, max
            if max_rep - min_rep > 2:
                mid = rng.randint(min_rep + 1, max_rep - 1)
                return rng.choice([min_rep, mid, max_rep])
            else:
                return rng.choice([min_rep, max_rep])
        
        # Pattern: {m} (exact count)
        elif min_rep == max_rep and min_rep > 0:
            if min_rep == 1:
                return 1
            elif min_rep == 2:
                return rng.choice([1, 2])  # underflow, exact
            else:
                mid = rng.randint(1, min_rep - 1)
                return rng.choice([1, mid, min_rep])  # underflow, middle, exact
        
        # Pattern: {m,} where m > 1 (unbounded from m)
        elif min_rep > 1 and max_rep >= MAX_LOOP_UNROLL:
            offset = rng.randint(0, 3)
            return rng.choice([min_rep, min_rep + offset])
        
        # Default: random within range
        else:
            if max_rep > min_rep:
                return rng.randint(min_rep, max_rep)
            else:
                return min_rep
    
    def _generate_bfs_sample(self, count):
        """
        Fallback: BFS-based sampling with random selection at each level.
        Used when stratified sampling doesn't generate enough samples.
        """
        import random
        from collections import deque
        
        results = set()
        queue = deque([(self.start_node, "", {}, {})])
        
        iterations = 0
        max_iterations = count * 100
        
        while queue and len(results) < count and iterations < max_iterations:
            iterations += 1
            
            # Process current level with random selection
            node, s, memory, active_caps = queue.popleft()
            
            next_states = self._get_next_states(node, s, memory, active_caps)
            
            # For class nodes, sample randomly instead of all
            if node.kind in ['class', 'any'] and len(next_states) > 5:
                next_states = random.sample(next_states, min(5, len(next_states)))
            
            for state in next_states:
                next_node, next_s, next_mem, next_active = state
                
                if next_node.kind == 'end':
                    clean_s = self._clean_conditional_markers(next_s, next_mem)
                    if clean_s is not None:
                        results.add(clean_s)
                else:
                    queue.append(state)
        
        return results
    
    def _clean_conditional_markers(self, s, memory):
        """
        Remove conditional markers from the string and validate conditions.
        Returns None if the conditional is invalid for the given memory state.
        """
        import re as re_std
        
        # Find all conditional markers in the string
        # Pattern: _COND{group_id}Y_{content} or _COND{group_id}N_{content}
        marker_pattern = r'_COND(\d+)([YN])_([^_]*)'
        
        result = s
        for match in re_std.finditer(marker_pattern, s):
            group_id = int(match.group(1))
            branch_type = match.group(2)  # 'Y' for yes branch, 'N' for no branch
            content = match.group(3)
            
            # Check if the group was captured
            group_captured = group_id in memory and memory[group_id]
            
            # Validate the conditional
            if branch_type == 'Y' and not group_captured:
                # Yes branch taken but group didn't match - invalid
                return None
            elif branch_type == 'N' and group_captured:
                # No branch taken but group did match - invalid
                return None
            
            # Replace the marker with just the content
            result = result.replace(match.group(0), content)
        
        return result

    def _select_relevant_violation_types(self):
        """
        Decide which violation types are interesting for the given regex.
        """
        has_unbounded = any(q in self.regex for q in ['*', '+'])
        # Matches {m,n} or {m}
        has_bounded = bool(re.search(r'\{\d+,?\d*\}', self.regex))
        
        has_anchors = any(n.kind == 'anchor' for n in self.nodes)
        has_backref = any(n.kind == 'backref' for n in self.nodes)
        has_class = any(n.kind in ['class', 'any'] for n in self.nodes)
        
        selected = []

        # 1. Add Lookaround/Conditional first if they exist
        if self.lookarounds:
            if any('lookahead' in l['type'] for l in self.lookarounds):
                selected.append('lookahead_violation')
            if any('lookbehind' in l['type'] for l in self.lookarounds):
                selected.append('lookbehind_violation')
        
        if hasattr(self, 'conditionals') and self.conditionals:
            selected.append('conditional_violation')

        # 2. Class-related
        if has_class:
            selected.extend(['character_class_violation', 'wrong_char_substitution'])

        # 3. Quantifiers: {m,n} gets above_max, others get below_min
        if has_unbounded or has_bounded:
            selected.append('quantifier_below_min')
        if has_bounded:
            selected.append('quantifier_above_max')

        # 4. Feature specific
        if has_anchors: selected.append('anchor_violation')
        if has_backref: selected.append('backreference_violation')

        # 5. Always useful structural mutations
        selected.extend(['missing_required_char', 'structure_violation'])

        return selected


    def generate_negative_samples(self, num_samples=10, *, seed=None, max_attempts=None, use_fullmatch=False):
        """
        Randomized negative sample generation.
        - Diverse outputs via randomized choices inside each violation generator
        - Validated to actually FAIL the regex
        - Optional seed for reproducibility
        """
        rng = random.Random(seed)

        negative_tests = set()

        violation_types = self._select_relevant_violation_types()
        print(f"Selected violation types for negative samples: {violation_types}")
        if not violation_types:
            violation_types = ['structure_violation', 'insert_invalid_char']

        # shuffle the order so we don't always cycle the same way
        rng.shuffle(violation_types)

        mapping = {
            'character_class_violation': lambda: self._generate_class_violations(rng=rng),
            'quantifier_below_min':      lambda: self._generate_quantifier_below_violations(rng=rng),
            'quantifier_above_max':      lambda: self._generate_quantifier_above_violations(rng=rng),
            'missing_required_char':     lambda: self._generate_missing_char_violations(rng=rng),
            'wrong_char_substitution':   lambda: self._generate_substitution_violations(rng=rng),
            'insert_invalid_char':       lambda: self._generate_insertion_violations(rng=rng),
            'delete_required_char':      lambda: self._generate_deletion_violations(rng=rng),
            'anchor_violation':          lambda: self._generate_anchor_violations(rng=rng),
            'structure_violation':       lambda: self._generate_structure_violations(rng=rng),
            'backreference_violation':   lambda: self._generate_backref_violations(rng=rng),
            'lookahead_violation':       lambda: self._generate_lookahead_violations(rng=rng),
            'lookbehind_violation':      lambda: self._generate_lookbehind_violations(rng=rng),
            'conditional_violation':     lambda: self._generate_conditional_violations(rng=rng),
        }

        if max_attempts is None:
            max_attempts = num_samples * 200  # higher since we randomize a lot

        def fails_regex(s: str) -> bool:
            try:
                # regex.fullmatch and regex.search are much more powerful 
                # and handle complex assertions that might crash standard 're'
                if use_fullmatch:
                    return regex.fullmatch(self.regex, s) is None
                return regex.search(self.regex, s) is None
            except regex.error:
                # If the string causes a computation error (timeout/limit), 
                # we treat it as a failure to validate (False)
                return False

        attempts = 0
        while len(negative_tests) < num_samples and attempts < max_attempts:
            attempts += 1

            # random type each attempt (not round-robin)
            v_type = rng.choice(violation_types)
            # print(f"Attempt {attempts}: Generating violation of type '{v_type}'")
            gen = mapping.get(v_type)
            if not gen:
                continue

            # each generator returns many candidates; sample a few to keep it moving
            candidates = gen() or []
            print(f"Attempt {attempts}: Generating violation of type '{v_type}', got {len(candidates)} candidates")
            if not candidates:
                continue

            rng.shuffle(candidates)
            for s in candidates[: rng.randint(3, 12)]:
                if len(negative_tests) >= num_samples:
                    break
                if s is None:
                    continue
                if fails_regex(s):
                    negative_tests.add(s)
                    break

        return rng.sample(sorted(negative_tests), k=min(num_samples, len(negative_tests)))


    def _get_valid_samples_cache(self, *, rng=None):
        if self.valid_samples_cache is None:
            # if your generate_sample supports randomness/seed, pass it through
            self.valid_samples_cache = self.generate_sample(50)
        return self.valid_samples_cache


    def _random_char_pool(self):
        # bigger pool than string.printable, but still “safe-ish”
        pool = []
        pool.extend(string.ascii_letters)
        pool.extend(string.digits)
        pool.extend(string.punctuation)
        pool.extend([" ", "\t", "\n"])
        # a few non-ascii for diversity (helps break ASCII-only classes)
        pool.extend(list("äöüßéèàçñøΩЖ中あ"))
        return pool


    def _get_char_outside_class(self, char_set, *, rng):
        # char_set might be set/list/str; normalize
        if char_set is None:
            char_set = set()
        if isinstance(char_set, str):
            char_set = set(char_set)
        else:
            char_set = set(char_set)

        pool = self._random_char_pool()
        outside = [c for c in pool if c not in char_set]
        if outside:
            return rng.choice(outside)

        # fallback: try random unicode range
        for _ in range(50):
            c = chr(rng.randint(0x20, 0x2FFF))
            if c not in char_set:
                return c
        return "!"


    def _random_outside_ascii_word(self, *, rng):
        # good for breaking \w / [A-Za-z0-9_]
        return rng.choice(["-", ".", " ", "\n", "中", "Ж", "Ω", "ä", "!", "@"])


    def _generate_class_violations(self, *, rng, max_nodes=1, per_node=1):
        violations = []

        class_nodes = [n for n in self.nodes if n.kind in ['class', 'any']]
        if not class_nodes:
            return violations

        rng.shuffle(class_nodes)
        for node in class_nodes[:max_nodes]:
            path = self._find_path_to_node(node)
            if not path:
                continue

            prefix = self._build_string_from_path(path[:-1], rng=rng)

            for _ in range(per_node):
                invalid_char = self._get_char_outside_class(node.payload, rng=rng)
                # also try adding random suffix noise
                suffix = ""
                if rng.random() < 0.6:
                    suffix = "".join(rng.choice(self._random_char_pool()) for _ in range(rng.randint(0, 4)))
                violations.append(prefix + invalid_char + suffix)

        return violations


    def _generate_quantifier_below_violations(self, *, rng, per_loop=1):
        violations = []
        loops = [n for n in self.nodes if n.kind == 'split' and 'Loop' in str(n.label)]
        rng.shuffle(loops)

        for node in loops:
            try:
                parts = str(node.label).split()
                if len(parts) <= 1:
                    continue
                range_part = parts[1].replace('?', '')
                min_val = int(range_part.split('-')[0])
                if min_val <= 0:
                    continue

                for _ in range(per_loop):
                    reps = rng.randint(0, max(0, min_val - 1))
                    s = self._build_string_with_repetitions(node, reps, rng=rng)
                    if s:
                        # sometimes truncate or add junk to break more
                        if rng.random() < 0.4 and len(s) > 0:
                            s = s[:-rng.randint(1, min(3, len(s)))]
                        if rng.random() < 0.3:
                            s += rng.choice(self._random_char_pool())
                        violations.append(s)
            except Exception:
                continue

        return violations


    def _generate_quantifier_above_violations(self, *, rng, per_loop=1):
        violations = []
        loops = [n for n in self.nodes if n.kind == 'split' and 'Loop' in str(n.label)]
        rng.shuffle(loops)

        for node in loops:
            try:
                parts = str(node.label).split()
                if len(parts) <= 1:
                    continue
                range_part = parts[1].replace('?', '')
                max_val = int(range_part.split('-')[1])

                # don’t explode
                cap = min(25, max_val + 10)
                for _ in range(per_loop):
                    reps = rng.randint(max_val + 1, cap) if max_val >= 0 else rng.randint(1, 10)
                    s = self._build_string_with_repetitions(node, reps, rng=rng)
                    if s:
                        # sometimes wrap with extra noise
                        if rng.random() < 0.5:
                            s = rng.choice(["X", "__", " "]) + s
                        if rng.random() < 0.5:
                            s = s + rng.choice(["Y", "__", "\n"])
                        violations.append(s)
            except Exception:
                continue

        return violations


    def _generate_missing_char_violations(self, *, rng, per_sample=1):
        violations = []
        valids = self._get_valid_samples_cache(rng=rng)
        if not valids:
            return violations

        samples = rng.sample(valids, k=min(10, len(valids)))
        for s in samples:
            if len(s) == 0:
                continue
            for _ in range(per_sample):
                if len(s) == 1:
                    violations.append("")
                    continue
                i = rng.randrange(len(s))
                violations.append(s[:i] + s[i+1:])
                # sometimes delete a slice
                if len(s) > 3 and rng.random() < 0.4:
                    a = rng.randrange(len(s))
                    b = rng.randrange(a, min(len(s), a + rng.randint(1, 3)))
                    violations.append(s[:a] + s[b:])

        return violations


    def _generate_substitution_violations(self, *, rng, per_sample=1):
        violations = []
        valids = self._get_valid_samples_cache(rng=rng)
        if not valids:
            return violations

        samples = rng.sample(valids, k=min(10, len(valids)))
        pool = self._random_char_pool()

        for s in samples:
            if not s:
                continue
            for _ in range(per_sample):
                i = rng.randrange(len(s))
                ch = s[i]

                # pick a “different type” char often
                if ch.isdigit():
                    new = rng.choice(list(string.ascii_letters) + ["_", "-", "中"])
                elif ch.isalpha():
                    new = rng.choice(list(string.digits) + ["@", "!", " "])
                elif ch == "_":
                    new = rng.choice(["-", ".", " ", "9"])
                else:
                    new = rng.choice(list(string.ascii_letters) + list(string.digits))

                if new == ch:
                    new = rng.choice(pool)

                t = s[:i] + new + s[i+1:]

                # sometimes change multiple positions
                if len(s) > 3 and rng.random() < 0.5:
                    j = rng.randrange(len(s))
                    if j != i:
                        t = t[:j] + rng.choice(pool) + t[j+1:]

                violations.append(t)

        return violations


    def _generate_insertion_violations(self, *, rng, per_sample=1):
        violations = []
        valids = self._get_valid_samples_cache(rng=rng)
        if not valids:
            return violations

        samples = rng.sample(valids, k=min(10, len(valids)))
        pool = self._random_char_pool()

        for s in samples:
            for _ in range(per_sample):
                pos = rng.randrange(len(s) + 1)
                ins_len = 1 if rng.random() < 0.7 else rng.randint(2, 5)
                blob = "".join(rng.choice(pool) for _ in range(ins_len))
                violations.append(s[:pos] + blob + s[pos:])

        return violations


    def _generate_deletion_violations(self, *, rng, per_sample=1):
        violations = []
        valids = self._get_valid_samples_cache(rng=rng)
        if not valids:
            return violations

        samples = rng.sample(valids, k=min(10, len(valids)))
        for s in samples:
            if len(s) <= 1:
                continue

            for _ in range(per_sample):
                a = rng.randrange(len(s))
                b = rng.randrange(a + 1, min(len(s) + 1, a + rng.randint(2, 6)))
                violations.append(s[:a] + s[b:])

            # also “thin out” with random mask
            if len(s) > 4:
                mask = [c for c in s if rng.random() < 0.5]
                violations.append("".join(mask))

        return violations


    def _generate_anchor_violations(self, *, rng, per_sample=1):
        violations = []
        has_start = any(n.kind == 'anchor' and n.payload == 'start' for n in self.nodes)
        has_end = any(n.kind == 'anchor' and n.payload == 'end' for n in self.nodes)

        valids = self._get_valid_samples_cache(rng=rng)
        if not valids:
            valids = ["a", "1", "test"]

        samples = rng.sample(valids, k=min(10, len(valids)))
        pool = self._random_char_pool()

        for s in samples:
            for _ in range(per_sample):
                pref = "".join(rng.choice(pool) for _ in range(rng.randint(1, 4)))
                suf = "".join(rng.choice(pool) for _ in range(rng.randint(1, 4)))

                if has_start and rng.random() < 0.7:
                    violations.append(pref + s)
                if has_end and rng.random() < 0.7:
                    violations.append(s + suf)

                # even without anchors, adding noise often breaks structure-based patterns
                if rng.random() < 0.3:
                    violations.append(pref + s + suf)

        return violations

    def _generate_structure_violations(self, *, rng, per_sample=1):
        violations = []
        valids = self._get_valid_samples_cache(rng=rng)
        if not valids:
            return violations

        samples = rng.sample(valids, k=min(10, len(valids)))
        pool = self._random_char_pool()

        for s in samples:
            if not s:
                violations.append(rng.choice(pool)) # Add noise to empty
                continue

            for _ in range(per_sample):
                mode = rng.choice(["reverse", "shuffle", "rotate", "dup_chunk", "drop_chunk", "noise"])
                
                if mode == "reverse":
                    violations.append(s[::-1])
                elif mode == "shuffle" and len(s) > 1:
                    chars = list(s)
                    rng.shuffle(chars)
                    violations.append("".join(chars))
                elif mode == "rotate" and len(s) > 1:
                    k = rng.randint(1, len(s))
                    violations.append(s[k:] + s[:k])
                elif mode == "dup_chunk" and len(s) >= 1:
                    a = rng.randrange(len(s))
                    # FIX: Ensure b is at least a + 1
                    b = rng.randint(a + 1, len(s)) 
                    chunk = s[a:b]
                    pos = rng.randrange(len(s) + 1)
                    violations.append(s[:pos] + chunk + s[pos:])
                elif mode == "drop_chunk" and len(s) > 1:
                    a = rng.randrange(len(s))
                    # FIX: Ensure b is at least a + 1
                    b = rng.randint(a + 1, len(s))
                    violations.append(s[:a] + s[b:])
                else:
                    # noise
                    t = list(s)
                    for _ in range(rng.randint(1, 3)):
                        i = rng.randint(0, len(t))
                        t.insert(i, rng.choice(pool))
                    violations.append("".join(t))

        return violations

    
    def _build_string_from_path(self, path, *, rng):
        out = []
        for node in path:
            if node.kind == 'match':
                out.append(node.payload)
            elif node.kind in ['class', 'any']:
                payload = node.payload
                if payload:
                    if isinstance(payload, str):
                        payload = list(payload)
                    else:
                        payload = list(payload)
                    out.append(rng.choice(payload))
                else:
                    out.append(rng.choice(self._random_char_pool()))
        return "".join(out)


    def _build_string_with_repetitions(self, loop_node, num_reps, *, rng):
        # still heuristic, but randomized so outputs diversify
        if not loop_node.next:
            return None

        # try to pick a representative token from loop body
        candidates = []
        for nxt in loop_node.next:
            if nxt.kind == 'match' and nxt.payload:
                candidates.append(nxt.payload)
            elif nxt.kind in ['class', 'any'] and nxt.payload:
                payload = list(nxt.payload) if isinstance(nxt.payload, str) else list(nxt.payload)
                if payload:
                    candidates.append(rng.choice(payload))

        if not candidates:
            candidates = [rng.choice(self._random_char_pool())]

        base = rng.choice(candidates)

        # sometimes vary per repetition instead of repeating same char
        if rng.random() < 0.5 and len(base) == 1:
            pool = candidates if len(candidates) > 1 else self._random_char_pool()
            return "".join(rng.choice(pool) for _ in range(num_reps))

        return base * num_reps

    def _generate_backref_violations(self, *, rng):
        violations = []
        valids = self._get_valid_samples_cache(rng=rng)
        samples = rng.sample(valids, k=min(5, len(valids)))
        for s in samples:
            if len(s) > 2:
                # Use rng instead of random.choice
                violations.append(s[:-1] + (rng.choice(string.ascii_letters)))
        return violations

    def _generate_lookahead_violations(self, *, rng):
        violations = []
        valids = self._get_valid_samples_cache(rng=rng)
        for la in self.lookarounds:
            if 'lookahead' in la['type']:
                content = la['content']
                for s in rng.sample(valids, k=min(3, len(valids))):
                    violations.append(s + "WRONG" + content)
        return violations

    def _generate_lookbehind_violations(self, *, rng):
        violations = []
        valids = self._get_valid_samples_cache(rng=rng)
        for lb in self.lookarounds:
            if 'lookbehind' in lb['type']:
                for s in rng.sample(valids, k=min(3, len(valids))):
                    content_len = len(lb['content'])
                    violations.append(s[content_len:] if len(s) > content_len else rng.choice(["X", "!", "9"]))
        return violations

    def _generate_conditional_violations(self, *, rng):
        valids = self._get_valid_samples_cache(rng=rng)
        samples = rng.sample(valids, k=min(5, len(valids)))
        # Diversify by reversing or shuffling
        return [s[::-1] if rng.random() > 0.5 else "".join(rng.sample(list(s), len(s))) for s in samples]

    def _find_path_to_node(self, target_node):
        """BFS to find a path from start to target node"""
        from collections import deque
        queue = deque([(self.start_node, [self.start_node])])
        visited = set()
        
        while queue:
            node, path = queue.popleft()
            if node == target_node:
                return path
            
            if node in visited:
                continue
            visited.add(node)
            
            for next_node in node.next:
                queue.append((next_node, path + [next_node]))
        
        return None
    
    # def generate_negative_samples(self, num_samples=10):
    #     """
    #     Improved negative sample generation with dynamic type selection
    #     and lookaround/conditional support.
    #     """
    #     negative_tests = set()
    #     # 1. Dynamically select what's relevant for this specific regex
    #     violation_types = self._select_relevant_violation_types()
    #     print(f"Selected violation types for negative samples: {violation_types}")
    #     if not violation_types:
    #         # Fallback to general mutations if nothing specific is detected
    #         violation_types = ['structure_violation', 'insert_invalid_char']

    #     violation_index = 0
    #     attempts = 0
    #     max_attempts = num_samples * 50
        
    #     while len(negative_tests) < num_samples and attempts < max_attempts:
    #         attempts += 1
    #         v_type = violation_types[violation_index % len(violation_types)]
            
    #         negatives = []
    #         # Map types to functions
    #         mapping = {
    #             'character_class_violation': self._generate_class_violations,
    #             'quantifier_below_min': self._generate_quantifier_below_violations,
    #             'quantifier_above_max': self._generate_quantifier_above_violations,
    #             'missing_required_char': self._generate_missing_char_violations,
    #             'wrong_char_substitution': self._generate_substitution_violations,
    #             'insert_invalid_char': self._generate_insertion_violations,
    #             'delete_required_char': self._generate_deletion_violations,
    #             'anchor_violation': self._generate_anchor_violations,
    #             'structure_violation': self._generate_structure_violations,
    #             'backreference_violation': self._generate_backref_violations,
    #             'lookahead_violation': self._generate_lookahead_violations,
    #             'lookbehind_violation': self._generate_lookbehind_violations,
    #             'conditional_violation': self._generate_conditional_violations
    #         }
            
    #         if v_type in mapping:
    #             negatives = mapping[v_type]()
            
    #         # 2. Add and Validate: Ensure they actually FAIL the regex
    #         import re as re_std
    #         for neg in negatives:
    #             if len(negative_tests) < num_samples:
    #                 # Use search or fullmatch based on your engine's default behavior
    #                 if not re_std.search(self.regex, neg):
    #                     negative_tests.add(neg)
            
    #         violation_index += 1
        
    #     return sorted(list(negative_tests))[:num_samples]

    # def _get_valid_samples_cache(self):
    #     """Get or generate valid samples for mutation-based violations"""
    #     if self.valid_samples_cache is None:
    #         self.valid_samples_cache = self.generate_sample(20)
    #     return self.valid_samples_cache

    # def _get_char_outside_class(self, char_set):
    #     """Get a character that's NOT in the given set"""
    #     all_chars = set(string.printable)
    #     outside = all_chars - char_set
    #     return list(outside)[0] if outside else '!'

    # def _generate_class_violations(self):
    #     """Violate character class constraints"""
    #     violations = []
        
    #     # Find all class nodes in the graph
    #     class_nodes = [n for n in self.nodes if n.kind in ['class', 'any']]
        
    #     for node in class_nodes[:3]:  # Limit to first 3 to avoid too many
    #         # Generate a path to this node
    #         path_to_node = self._find_path_to_node(node)
    #         if path_to_node:
    #             # Build string up to this node
    #             prefix = self._build_string_from_path(path_to_node[:-1])
    #             # Use invalid char at this node
    #             invalid_char = self._get_char_outside_class(node.payload)
    #             violations.append(prefix + invalid_char)
        
    #     return violations

    # def _generate_quantifier_below_violations(self):
    #     """Generate strings with repetitions below minimum"""
    #     violations = []
        
    #     # Find split nodes that represent loops
    #     for node in self.nodes:
    #         if node.kind == 'split' and 'Loop' in node.label:
    #             # Extract min from label like "Loop 1-3"
    #             try:
    #                 parts = node.label.split()
    #                 if len(parts) > 1:
    #                     range_part = parts[1].replace('?', '')
    #                     min_val = int(range_part.split('-')[0])
                        
    #                     if min_val > 0:
    #                         # Generate with min-1 repetitions
    #                         # Find what this loop repeats
    #                         violation = self._build_string_with_repetitions(node, min_val - 1)
    #                         if violation:
    #                             violations.append(violation)
    #             except:
    #                 pass
        
    #     return violations

    # def _generate_quantifier_above_violations(self):
    #     """Generate strings with repetitions above maximum"""
    #     violations = []
        
    #     for node in self.nodes:
    #         if node.kind == 'split' and 'Loop' in node.label:
    #             try:
    #                 parts = node.label.split()
    #                 if len(parts) > 1:
    #                     range_part = parts[1].replace('?', '')
    #                     max_val = int(range_part.split('-')[1])
                        
    #                     # Generate with max+1 repetitions (but not if max was already capped)
    #                     if max_val < 10:  # Don't go crazy
    #                         violation = self._build_string_with_repetitions(node, max_val + 1)
    #                         if violation:
    #                             violations.append(violation)
    #             except:
    #                 pass
        
    #     return violations

    # def _generate_missing_char_violations(self):
    #     """Delete required characters from valid strings"""
    #     violations = []
    #     valid_samples = self._get_valid_samples_cache()
        
    #     for sample in valid_samples[:3]:
    #         if len(sample) > 1:
    #             # Delete first required char
    #             violations.append(sample[1:])
    #             # Delete last required char
    #             violations.append(sample[:-1])
    #             # Delete middle char
    #             if len(sample) > 2:
    #                 mid = len(sample) // 2
    #                 violations.append(sample[:mid] + sample[mid+1:])
        
    #     return violations

    # def _generate_substitution_violations(self):
    #     """Substitute correct characters with wrong types"""
    #     violations = []
    #     valid_samples = self._get_valid_samples_cache()
        
    #     for sample in valid_samples[:3]:
    #         for i in range(len(sample)):
    #             char = sample[i]
    #             # Substitute with wrong type
    #             if char.isalpha():
    #                 wrong_char = '0' if char.islower() else '9'
    #             elif char.isdigit():
    #                 wrong_char = 'X'
    #             elif char in '()[]{}':
    #                 wrong_char = 'a'
    #             else:
    #                 wrong_char = '@'
                
    #             violations.append(sample[:i] + wrong_char + sample[i+1:])
        
    #     return violations

    # def _generate_insertion_violations(self):
    #     """Insert invalid characters into valid strings"""
    #     violations = []
    #     valid_samples = self._get_valid_samples_cache()
        
    #     invalid_chars = ['!', '@', '#', 'X', '9', ' ', '\n']
        
    #     for sample in valid_samples[:3]:
    #         for insert_pos in [0, len(sample)//2, len(sample)]:
    #             for inv_char in invalid_chars[:2]:  # Use first 2
    #                 violations.append(sample[:insert_pos] + inv_char + sample[insert_pos:])
        
    #     return violations

    # def _generate_deletion_violations(self):
    #     """Delete characters from various positions"""
    #     violations = []
    #     valid_samples = self._get_valid_samples_cache()
        
    #     for sample in valid_samples[:3]:
    #         # Delete every other character
    #         if len(sample) > 3:
    #             deleted = ''.join(sample[i] for i in range(0, len(sample), 2))
    #             violations.append(deleted)
        
    #     return violations

    # def _generate_anchor_violations(self):
    #     """Violate anchor constraints (^, $)"""
    #     violations = []
    #     has_start_anchor = any(n.kind == 'anchor' and n.payload == 'start' for n in self.nodes)
    #     has_end_anchor = any(n.kind == 'anchor' and n.payload == 'end' for n in self.nodes)
        
    #     valid_samples = self._get_valid_samples_cache()
        
    #     for sample in valid_samples[:2]:
    #         if has_start_anchor:
    #             # Add prefix (violates ^)
    #             violations.append('X' + sample)
    #             violations.append('__' + sample)
            
    #         if has_end_anchor:
    #             # Add suffix (violates $)
    #             violations.append(sample + 'Y')
    #             violations.append(sample + '__')
        
    #     return violations

    # def _generate_structure_violations(self):
    #     """Violate overall structure"""
    #     violations = []
    #     valid_samples = self._get_valid_samples_cache()
        
    #     for sample in valid_samples[:2]:
    #         # Reverse the string
    #         violations.append(sample[::-1])
            
    #         # Shuffle characters
    #         if len(sample) > 2:
    #             chars = list(sample)
    #             # Simple shuffle: swap first and last half
    #             mid = len(chars) // 2
    #             shuffled = chars[mid:] + chars[:mid]
    #             violations.append(''.join(shuffled))
        
    #     return violations

    # def _generate_backref_violations(self):
    #     """
    #     Violate backreference constraints in sophisticated ways.
    #     For patterns like (a)\1 or ([ab])\1, the backreference must match exactly.
    #     We generate violations by:
    #     1. Changing characters in the backreference part
    #     2. Using different case (if letters)
    #     3. Using similar but different characters
    #     4. Partial matches
    #     5. Swapping captured group with backreference
    #     """
    #     violations = []
        
    #     # Find backreference nodes
    #     backref_nodes = [n for n in self.nodes if n.kind == 'backref']
    #     if not backref_nodes:
    #         return violations
        
    #     # Find corresponding capture groups
    #     group_starts = {n.payload: n for n in self.nodes if n.kind == 'group_start'}
        
    #     for backref_node in backref_nodes:
    #         group_id = backref_node.payload
            
    #         # Strategy 1: Build valid string then modify backreference part
    #         valid_samples = self._get_valid_samples_cache()
    #         for sample in valid_samples[:3]:
    #             if len(sample) < 2:
    #                 continue
                
    #             # Find where the backreference likely is (second half usually)
    #             # This is heuristic - proper implementation would track positions
    #             mid = len(sample) // 2
                
    #             # Violation 1a: Change one character in backreference
    #             if mid < len(sample):
    #                 char_to_change = sample[mid]
    #                 if char_to_change.isalpha():
    #                     new_char = char_to_change.swapcase()  # Change case
    #                 elif char_to_change.isdigit():
    #                     new_char = str((int(char_to_change) + 1) % 10)  # Different digit
    #                 else:
    #                     new_char = 'X'
    #                 violations.append(sample[:mid] + new_char + sample[mid+1:])
                
    #             # Violation 1b: Use similar character (a->b, 1->2)
    #             if mid < len(sample):
    #                 similar_map = {
    #                     'a': 'b', 'b': 'c', 'c': 'a',
    #                     '0': '1', '1': '2', '2': '0',
    #                     '(': ')', ')': '('
    #                 }
    #                 char_to_change = sample[mid]
    #                 new_char = similar_map.get(char_to_change, 'X')
    #                 violations.append(sample[:mid] + new_char + sample[mid+1:])
            
    #         # Strategy 2: Graph-based approach - find paths through capture and backref
    #         group_start = group_starts.get(group_id)
    #         if group_start:
    #             # Build string with capture group
    #             path_to_group = self._find_path_to_node(group_start)
    #             if path_to_group:
    #                 # Build prefix before group
    #                 prefix = self._build_string_from_path(path_to_group[:-1])
                    
    #                 # Find what the group captures (walk from group_start to group_end)
    #                 captured_content = self._extract_group_content(group_start, group_id)
                    
    #                 if captured_content:
    #                     # Now find path to backref node
    #                     path_to_backref = self._find_path_from_node_to_node(group_start, backref_node)
    #                     if path_to_backref:
    #                         middle = self._build_string_from_path(path_to_backref[:-1])
                            
    #                         # Generate violations
    #                         for cap in captured_content[:2]:
    #                             # Violation 2a: Different length
    #                             if len(cap) > 1:
    #                                 violations.append(prefix + cap + middle + cap[:-1])
    #                                 violations.append(prefix + cap + middle + cap + 'X')
                                
    #                             # Violation 2b: Completely different content
    #                             different = self._generate_different_string(cap)
    #                             violations.append(prefix + cap + middle + different)
                                
    #                             # Violation 2c: Reversed content
    #                             violations.append(prefix + cap + middle + cap[::-1])
                                
    #                             # Violation 2d: Empty backreference (if cap is not empty)
    #                             if cap:
    #                                 violations.append(prefix + cap + middle)
                                
    #                             # Violation 2e: Partial match (first half matches)
    #                             if len(cap) > 1:
    #                                 violations.append(prefix + cap + middle + cap[:len(cap)//2] + 'X')
        
    #     return violations
    
    # def _extract_group_content(self, group_start_node, group_id):
    #     """
    #     Extract possible content between group_start and group_end nodes.
    #     Returns a list of possible captured strings.
    #     """
    #     from collections import deque
        
    #     # Find the group_end node for this group
    #     group_end = None
    #     for n in self.nodes:
    #         if n.kind == 'group_end' and n.payload == group_id:
    #             group_end = n
    #             break
        
    #     if not group_end:
    #         return []
        
    #     # BFS from group_start to group_end, collecting possible strings
    #     queue = deque([(group_start_node, "")])
    #     visited = set()
    #     results = []
    #     max_results = 5
        
    #     while queue and len(results) < max_results:
    #         node, current_string = queue.popleft()
            
    #         if (node, current_string) in visited:
    #             continue
    #         visited.add((node, current_string))
            
    #         if node == group_end:
    #             results.append(current_string)
    #             continue
            
    #         for next_node in node.next:
    #             next_string = current_string
                
    #             if next_node.kind == 'match':
    #                 next_string += next_node.payload
    #             elif next_node.kind in ['class', 'any']:
    #                 if next_node.payload:
    #                     # Take first few chars from class
    #                     for char in list(next_node.payload)[:2]:
    #                         queue.append((next_node, current_string + char))
    #                     continue
                
    #             queue.append((next_node, next_string))
        
    #     return results if results else ['a']  # Default fallback
    
    # def _find_path_from_node_to_node(self, start_node, target_node):
    #     """BFS to find path from start_node to target_node"""
    #     from collections import deque
    #     queue = deque([(start_node, [start_node])])
    #     visited = set()
        
    #     while queue:
    #         node, path = queue.popleft()
    #         if node == target_node:
    #             return path
            
    #         if node in visited:
    #             continue
    #         visited.add(node)
            
    #         for next_node in node.next:
    #             queue.append((next_node, path + [next_node]))
        
    #     return None
    
    # def _generate_different_string(self, original):
    #     """Generate a string different from original but similar length"""
    #     if not original:
    #         return 'X'
        
    #     # Strategy: flip each character to something different
    #     result = []
    #     for char in original:
    #         if char.isalpha():
    #             result.append('z' if char == 'a' else 'a')
    #         elif char.isdigit():
    #             result.append('9' if char == '0' else '0')
    #         elif char == '(':
    #             result.append(')')
    #         elif char == ')':
    #             result.append('(')
    #         else:
    #             result.append('X')
    #     return ''.join(result)
    
    # def _generate_lookahead_violations(self):
    #     """
    #     Violate lookahead assertions.
        
    #     Positive lookahead (?=...): Assert that pattern CAN match ahead
    #     - Violation: Generate strings where the lookahead pattern does NOT match
        
    #     Negative lookahead (?!...): Assert that pattern CANNOT match ahead  
    #     - Violation: Generate strings where the lookahead pattern DOES match
    #     """
    #     violations = []
        
    #     if not self.lookarounds:
    #         return violations
        
    #     valid_samples = self._get_valid_samples_cache()
        
    #     for lookaround in self.lookarounds:
    #         if 'lookahead' not in lookaround['type']:
    #             continue
            
    #         la_type = lookaround['type']
    #         content = lookaround['content']
            
    #         if la_type == 'positive_lookahead':
    #             # For \d(?=px), generate digits NOT followed by "px"
    #             # Strategy: Build valid prefix, then violate the lookahead
    #             for sample in valid_samples[:3]:
    #                 # Add wrong suffix
    #                 violations.append(sample + 'xx')
    #                 violations.append(sample + 'py')
    #                 violations.append(sample + 'p')
    #                 violations.append(sample + 'x')
    #                 violations.append(sample)  # No suffix at all
                
    #             # Generate specific violations based on lookahead content
    #             # Parse lookahead content to understand what it expects
    #             if content.isalpha():
    #                 # Simple string lookahead
    #                 violations.append('5' + 'X' + content[1:])  # Wrong first char
    #                 violations.append('5' + content[:-1])  # Truncated
    #                 violations.append('5' + content[::-1])  # Reversed
                
    #         elif la_type == 'negative_lookahead':
    #             # For \d(?!abc), generate digits followed by "abc"
    #             # Strategy: Include the forbidden pattern
    #             for sample in valid_samples[:3]:
    #                 # Add the forbidden pattern
    #                 violations.append(sample + content)
    #                 violations.append(sample + content + 'x')
                
    #             # Generate with forbidden pattern at various positions
    #             violations.append('5' + content)
    #             violations.append('9' + content)
        
    #     return violations
    
    # def _generate_lookbehind_violations(self):
    #     """
    #     Violate lookbehind assertions.
        
    #     Positive lookbehind (?<=...): Assert that pattern CAN match behind
    #     - Violation: Generate strings where the lookbehind pattern does NOT match
        
    #     Negative lookbehind (?<!...): Assert that pattern CANNOT match behind
    #     - Violation: Generate strings where the lookbehind pattern DOES match
    #     """
    #     violations = []
        
    #     if not self.lookarounds:
    #         return violations
        
    #     valid_samples = self._get_valid_samples_cache()
        
    #     for lookaround in self.lookarounds:
    #         if 'lookbehind' not in lookaround['type']:
    #             continue
            
    #         lb_type = lookaround['type']
    #         content = lookaround['content']
            
    #         if lb_type == 'positive_lookbehind':
    #             # For (?<=@)\w+, generate words NOT preceded by "@"
    #             # Strategy: Build without the required prefix
    #             for sample in valid_samples[:3]:
    #                 # Remove or change prefix
    #                 violations.append(sample)  # No prefix
    #                 violations.append('#' + sample)  # Wrong prefix
    #                 violations.append('!' + sample)  # Wrong prefix
                
    #             # Generate with wrong prefix
    #             if content:
    #                 violations.append('X' + 'test')
    #                 violations.append('Y' + 'word')
                
    #         elif lb_type == 'negative_lookbehind':
    #             # For (?<!@)\w+, generate words preceded by "@"
    #             # Strategy: Add the forbidden prefix
    #             for sample in valid_samples[:3]:
    #                 violations.append(content + sample)
                
    #             # Generate with forbidden prefix
    #             violations.append(content + 'test')
    #             violations.append(content + 'word')
        
    #     return violations
    
    # def _generate_conditional_violations(self):
    #     """
    #     Violate conditional pattern logic.
        
    #     For (a)?(?(1)b|c):
    #     - Valid: "ab" (group 1 matched, took yes branch)
    #     - Valid: "c" (group 1 didn't match, took no branch)
    #     - Invalid: "ac" (group 1 matched, took no branch)
    #     - Invalid: "b" (group 1 didn't match, took yes branch)
    #     """
    #     violations = []
        
    #     if not self.conditionals:
    #         return violations
        
    #     # Analyze the original regex to understand conditional structure
    #     import re as re_std
        
    #     # Match (a)?(?(1)b|c) pattern
    #     pattern = r'\(([^)]+)\)\?\(\?\((\d+)\)([^|)]+)(?:\|([^)]+))?\)'
    #     matches = re_std.finditer(pattern, self.regex)
        
    #     for match in matches:
    #         optional_group = match.group(1)
    #         group_id = match.group(2)
    #         yes_branch = match.group(3)
    #         no_branch = match.group(4) if match.group(4) else ''
            
    #         # Generate violations
    #         # Violation 1: Group matched but took no branch
    #         if no_branch:
    #             violations.append(optional_group + no_branch)
    #             violations.append(optional_group + optional_group + no_branch)
            
    #         # Violation 2: Group didn't match but took yes branch
    #         violations.append(yes_branch)
    #         violations.append('X' + yes_branch)
            
    #         # Violation 3: Neither branch
    #         violations.append(optional_group)
    #         violations.append('')
            
    #         # Violation 4: Both branches
    #         if no_branch:
    #             violations.append(optional_group + yes_branch + no_branch)
    #             violations.append(yes_branch + no_branch)
        
    #     return violations

    # def _find_path_to_node(self, target_node):
    #     """BFS to find a path from start to target node"""
    #     from collections import deque
    #     queue = deque([(self.start_node, [self.start_node])])
    #     visited = set()
        
    #     while queue:
    #         node, path = queue.popleft()
    #         if node == target_node:
    #             return path
            
    #         if node in visited:
    #             continue
    #         visited.add(node)
            
    #         for next_node in node.next:
    #             queue.append((next_node, path + [next_node]))
        
    #     return None

    # def _build_string_from_path(self, path):
    #     """Build a string by following a path through the graph"""
    #     result = ""
    #     for node in path:
    #         if node.kind == 'match':
    #             result += node.payload
    #         elif node.kind in ['class', 'any']:
    #             # Pick first char from set
    #             if node.payload:
    #                 result += list(node.payload)[0]
    #     return result

    # def _build_string_with_repetitions(self, loop_node, num_reps):
    #     """Build a string with specific number of repetitions"""
    #     # Find what gets repeated in this loop
    #     # This is a simplified version - full implementation would trace loop body
        
    #     # Find a simple path through the loop
    #     if loop_node.next:
    #         base_char = 'a'  # Default
    #         for next_node in loop_node.next:
    #             if next_node.kind == 'match':
    #                 base_char = next_node.payload
    #                 break
    #             elif next_node.kind in ['class', 'any']:
    #                 base_char = list(next_node.payload)[0] if next_node.payload else 'a'
    #                 break
            
    #         return base_char * num_reps
        
    #     return None

# --- DEMO ---
if __name__ == "__main__":
    
    print("=" * 60)
    print("Example 1: Simple Character Classes")
    regex1 = r"[a-c]\d{2}"
    builder1 = RegexGraphBuilder(regex1)
    count1, has_backref1 = builder1.count_paths()
    print(f"Regex: {regex1}")
    print(f"Total possible strings: {count1}")
    samples1 = builder1.generate_sample(max_samples=count1)
    # print(f"Generated {len(samples1)} samples: {samples1}")
    
    print("\n" + "=" * 60)
    print("Example 2: Small Set - Exhaustive Generation")
    regex2 = r"[abc][12]"
    builder2 = RegexGraphBuilder(regex2)
    count2, has_backref2 = builder2.count_paths()
    print(f"Regex: {regex2}")
    print(f"Total possible strings: {count2}")
    samples2 = builder2.generate_sample()
    print(f"Generated ALL {len(samples2)} strings: {samples2}")
    
    print("\n" + "=" * 60)
    print("Example 3: Large Set - Stratified Sampling")
    regex3 = r"[a-z]{3}\d{2}"
    builder3 = RegexGraphBuilder(regex3)
    count3, has_backref3 = builder3.count_paths()
    print(f"Regex: {regex3}")
    print(f"Total possible strings: {count3:,}")
    
    import time
    start_time = time.perf_counter()
    samples3 = builder3.generate_sample(max_samples=count3)
    elapsed_time = time.perf_counter() - start_time
    
    print(f"Sampled {len(samples3)} diverse strings: {samples3[:10]}")
    print(f"Time taken: {elapsed_time:.4f} seconds")
    
    # print("\n" + "=" * 60)
    # print("Example 4: Medium Set - Exhaustive vs Sampling")
    # regex4 = r"[a-d]{2}\d"
    # builder4 = RegexGraphBuilder(regex4)
    # count4, has_backref4 = builder4.count_paths()
    # print(f"Regex: {regex4}")
    # print(f"Total possible strings: {count4}")
    # samples4 = builder4.generate_sample()  # Should generate all
    # print(f"Generated {len(samples4)} strings")
    # print(f"First 20: {samples4[:20]}")
    # print(f"Last 20: {samples4[-20:]}")
    
    # print("\n" + "=" * 60)
    # print("Example 5: Recursive Pattern (Balanced Parentheses)")
    # regex5 = r'\((?:[^()]|(?R))*\)'
    # print(f"Regex: {regex5}")
    # print(f"Recursion expanded to depth {MAX_RECURSION_DEPTH}")
    # builder5 = RegexGraphBuilder(regex5)
    # count5, has_backref5 = builder5.count_paths()
    # print(f"Total possible strings (up to depth {MAX_RECURSION_DEPTH}): {count5}")
    # samples5 = builder5.generate_sample(max_samples=20)
    # print(f"Sample strings: {samples5[:20]}")
    
    # print("\n" + "=" * 60)
    # print("Example 6: Word Characters")
    # regex6 = r"\w{2,3}"
    # builder6 = RegexGraphBuilder(regex6)
    # count6, has_backref6 = builder6.count_paths()
    # print(f"Regex: {regex6}")
    # print(f"Total possible strings: {count6:,}")
    # samples6 = builder6.generate_sample(max_samples=50)
    # print(f"Sampled {len(samples6)} strings: {samples6[:10]}")
    
    # print("\n" + "=" * 60)
    # print("Example 7: Complex with Alternation")
    # regex7 = r"(cat|dog)\d"
    # builder7 = RegexGraphBuilder(regex7)
    # count7, has_backref7 = builder7.count_paths()
    # print(f"Regex: {regex7}")
    # print(f"Total possible strings: {count7}")
    # samples7 = builder7.generate_sample()
    # print(f"All {len(samples7)} strings: {samples7}")
    
    # print("\n" + "=" * 60)
    # print("Example 8: Unicode Properties")
    # regex8 = r"\p{L}+"
    # builder8 = RegexGraphBuilder(regex8)
    # count8, has_backref8 = builder8.count_paths()
    # print(f"Regex: {regex8}")
    # print(f"Expanded to: [a-zA-Z...]+ (Latin letters)")
    # print(f"Total possible strings: {count8:,}")
    # samples8 = builder8.generate_sample(max_samples=10)
    # print(f"Sample strings: {samples8[:10]}")
    
    # Demo negative test generation
    print("\n" + "=" * 60)
    print("Example 6: Negative Test Generation")
    regex6 = r"[a-z]{2,4}\d+"
    builder6 = RegexGraphBuilder(regex6)
    print(f"Regex: {regex6}")
    print(f"Description: 2-4 lowercase letters followed by 1+ digits")
    
    # Generate positive samples
    positive = builder6.generate_sample(5)
    print(f"\nPositive samples (SHOULD match):")
    for p in positive:
        print(f"  ✓ '{p}'")
    
    # Generate negative samples
    num_negative = 150
    negative = builder6.generate_negative_samples(num_negative)
    print(f"\nNegative samples (should NOT match) - requested {num_negative}:")
    for n in negative[:num_negative]:
        print(f"  ✗ '{n}'")
    
    # Verify with actual regex
    import re
    pattern = re.compile(regex6)
    print(f"\nVerification:")
    matches = sum(1 for n in negative if pattern.fullmatch(n))
    print(f"  Negative samples that incorrectly match: {matches}/{len(negative)}")
    print(f"  Success rate: {((len(negative) - matches) / len(negative) * 100):.1f}%")
    
    # # Visualize the recursive pattern
    # print("\n" + "=" * 60)
    # print("Generating graph visualization for recursive pattern...")
    # builder2.visualize()