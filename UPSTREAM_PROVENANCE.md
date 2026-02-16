# OSRL Upstream Provenance

This folder is now a full clone of OSRL with local patches applied.

- Upstream project: `https://github.com/liuzuxin/OSRL`
- Local clone path: `third_party/osrl_patched`
- Upstream remote in this clone: `origin`
- Upstream branch tracked here: `origin/main`
- Upstream commit pinned: `03a4586463690566ec8d7830ca4027b46d4ad02f`
- Upstream commit date: `2024-09-13T10:01:21-07:00`
- Upstream license: `Apache-2.0` (from `origin/main:LICENSE`)

## Why this exists

We used to have a trimmed vendor snapshot with only selected files. This clone
keeps the whole upstream tree so git can show exactly what is original and what
was changed locally.

## Local file status vs upstream commit

Status meaning:
- `unchanged`: local blob hash equals `origin/main:<path>`
- `patched`: local file exists upstream but content differs
- `added`: local file does not exist at the same upstream path

| File | Status | Local blob | Upstream blob |
| --- | --- | --- | --- |
| `osrl/__init__.py` | `unchanged` | `c0802f36da91adc9c8cfd11bc27d959f87e5810e` | `c0802f36da91adc9c8cfd11bc27d959f87e5810e` |
| `osrl/algorithms/__init__.py` | `patched` | `668318286b7e0c87702f9b594f1404a753440a6e` | `59d62fe590f9d684b2708a2e1739e6ce447389d0` |
| `osrl/algorithms/bc.py` | `unchanged` | `aa903af26d106d475f937045269faf2119d970af` | `aa903af26d106d475f937045269faf2119d970af` |
| `osrl/algorithms/bcql.py` | `unchanged` | `d63b3ea27bf3c1d971a35ab2bdf91af258fd25d5` | `d63b3ea27bf3c1d971a35ab2bdf91af258fd25d5` |
| `osrl/algorithms/bearl.py` | `unchanged` | `3916f533c3e343d6d99e02898914c54657e7be8a` | `3916f533c3e343d6d99e02898914c54657e7be8a` |
| `osrl/algorithms/cdt.py` | `patched` | `6231d891ab098a4c9c739a903ffd2e6eb2ce3b89` | `f384be19ef08f7cd6bdabda711d26121615548e4` |
| `osrl/algorithms/coptidice.py` | `unchanged` | `0ddd8208ce09a216e134d8e0efc9940f155df220` | `0ddd8208ce09a216e134d8e0efc9940f155df220` |
| `osrl/algorithms/cpq.py` | `unchanged` | `f1d42cadb59625d8484f352a25ddfdf7770d0bb1` | `f1d42cadb59625d8484f352a25ddfdf7770d0bb1` |
| `osrl/common/__init__.py` | `patched` | `645aba6f7ebf9a7238e956f90acbcf8f67b59662` | `182fee6ad765f578f40b3012181c23261fa98d85` |
| `osrl/common/dataset.py` | `unchanged` | `a0e379b7a1301f3fa4f239ad63253e7b5b2b475e` | `a0e379b7a1301f3fa4f239ad63253e7b5b2b475e` |
| `osrl/common/exp_util.py` | `unchanged` | `aa877b743248f93c07f7b7c690cae4c8f0616852` | `aa877b743248f93c07f7b7c690cae4c8f0616852` |
| `osrl/common/net.py` | `unchanged` | `f267d2437169ff55298689380cc20f2f52ec182f` | `f267d2437169ff55298689380cc20f2f52ec182f` |

## Commands used for migration

```powershell
git clone https://github.com/liuzuxin/OSRL.git third_party\osrl_patched
```

Then local patched files were copied into this full clone, and the old subset
folder was removed.

## Recompute status table later

```powershell
$repo='third_party\osrl_patched'
$base=(Resolve-Path $repo).Path
$files=Get-ChildItem "$repo\osrl" -Recurse -File -Filter *.py | Sort-Object FullName
foreach($item in $files){
  $rel=$item.FullName.Substring($base.Length+1).Replace('\','/')
  $localBlob=git -C $repo hash-object -- $rel
  git -C $repo cat-file -e "origin/main:$rel" 2>$null
  if($LASTEXITCODE -eq 0){
    $upstreamBlob=git -C $repo rev-parse "origin/main:$rel"
    $status=if($localBlob -eq $upstreamBlob){'unchanged'}else{'patched'}
  } else {
    $upstreamBlob=''
    $status='added'
  }
  [PSCustomObject]@{File=$rel;Status=$status;LocalBlob=$localBlob;UpstreamBlob=$upstreamBlob}
}
```
