# OA one-click bot extractor (bookmarklet)
Drag this to your bookmarks bar; click it on any OA bot page — copies the
page's bot config as JSON to your clipboard. Paste into an "[oa-import]" issue.

javascript:(()=>{const t=document.title;const b=document.body.innerText;const j={name:t.replace(/\s*[|–-]\s*Option Alpha.*/i,"").trim(),source_url:location.href,raw_text:b.slice(0,4000)};navigator.clipboard.writeText(JSON.stringify(j,null,2)).then(()=>alert("Bot JSON copied — paste into an [oa-import] GitHub issue"))})();

The raw_text carries every visible setting (legs/delta/DTE/exits); the
importer (or an employee) parses it into the template schema.
