# save_verbatim.ps1 — Stop hook (canonical template; copied into each project's .claude/hooks/)
# Appends Claude's verbatim assistant text from the just-finished turn to
# verbatim_history/<YYYY-MM-DD>_verbatim.md. Reads the hook input JSON on stdin
# (which carries transcript_path), parses the JSONL transcript, finds the last
# genuine user prompt, and collects every assistant text block after it.
# Portable: the output folder is resolved relative to THIS script (../../verbatim_history),
# so no per-project edits are needed. Best-effort: any failure exits 0 silently
# so it never blocks the session.

$ErrorActionPreference = 'Stop'
try {
    $raw = [Console]::In.ReadToEnd()
    if (-not $raw) { exit 0 }

    $hook = $raw | ConvertFrom-Json
    $tp = $hook.transcript_path
    if (-not $tp -or -not (Test-Path -LiteralPath $tp)) { exit 0 }

    $outDir = Join-Path $PSScriptRoot '..\..\verbatim_history'
    if (-not (Test-Path -LiteralPath $outDir)) {
        New-Item -ItemType Directory -Path $outDir -Force | Out-Null
    }

    # Parse the JSONL transcript (skip any unparseable lines).
    # -Encoding UTF8 is required: PS 5.1 Get-Content defaults to ANSI and would
    # corrupt non-ASCII characters (em-dashes, accents) in the assistant text.
    $events = foreach ($line in (Get-Content -LiteralPath $tp -Encoding UTF8)) {
        if (-not $line.Trim()) { continue }
        try { $line | ConvertFrom-Json } catch { }
    }
    if (-not $events) { exit 0 }

    # Find the index of the last GENUINE user prompt: a type=user event whose
    # content is a plain string, or an array containing a 'text' block. This
    # excludes tool_result-only user events injected by the harness.
    $startIdx = 0
    for ($i = $events.Count - 1; $i -ge 0; $i--) {
        $e = $events[$i]
        if ($e.type -ne 'user') { continue }
        $c = $e.message.content
        $isPrompt = $false
        if ($c -is [string]) {
            $isPrompt = $true
        } elseif ($c) {
            foreach ($b in $c) { if ($b.type -eq 'text') { $isPrompt = $true; break } }
        }
        if ($isPrompt) { $startIdx = $i; break }
    }

    # Collect every assistant text block from that prompt forward.
    $texts = New-Object System.Collections.Generic.List[string]
    for ($i = $startIdx; $i -lt $events.Count; $i++) {
        $e = $events[$i]
        if ($e.type -ne 'assistant') { continue }
        $c = $e.message.content
        if (-not $c) { continue }
        foreach ($b in $c) {
            if ($b.type -eq 'text' -and $b.text) { $texts.Add($b.text) }
        }
    }
    if ($texts.Count -eq 0) { exit 0 }

    $stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    $day = Get-Date -Format 'yyyy-MM-dd'
    $outFile = Join-Path $outDir ("{0}_verbatim.md" -f $day)
    $body = [string]::Join("`n`n", $texts)
    $entry = "`n---`n## $stamp`n`n$body`n"
    Add-Content -LiteralPath $outFile -Value $entry -Encoding UTF8
} catch { }
exit 0
