import { execSync } from "child_process";
import { readdirSync, statSync, readFileSync } from "fs";
import path from "path";

//
// Detect FastAPI folders by looking for:
// - pyproject.toml containing "fastapi"
// - requirements.txt containing "fastapi"
// - main.py or app/main.py importing fastapi
//
function isFastAPIFolder(dir) {
    const files = readdirSync(dir);

    // pyproject.toml
    if (files.includes("pyproject.toml")) {
        const txt = readFileSync(path.join(dir, "pyproject.toml"), "utf8");
        if (txt.includes("fastapi")) return true;
    }

    // requirements.txt
    if (files.includes("requirements.txt")) {
        const txt = readFileSync(path.join(dir, "requirements.txt"), "utf8");
        if (txt.includes("fastapi")) return true;
    }

    return false;
}

//
// Recursively find all FastAPI folders
//
function findFastAPIFolders(startDir) {
    const results = [];

    function walk(dir) {
        const files = readdirSync(dir);

        if (isFastAPIFolder(dir)) {
            results.push(dir);
        }

        for (const file of files) {
            const full = path.join(dir, file);
            if (statSync(full).isDirectory()) {
                if (file !== "node_modules" && file !== "__pycache__") {
                    walk(full);
                }
            }
        }
    }

    walk(startDir);
    return results;
}

//
// Run FastAPI export step in each backend
//
function buildFastAPIBackends(folders) {
    for (const backend of folders) {
        console.log("Running FastAPI export in:", backend);

        // Adjust this command to your actual export step
        execSync("python generate_api.py", {
            cwd: backend,
            stdio: "inherit"
        });

        console.log("FastAPI export complete:", backend);
    }
}

//
// Recursively build Node packages AFTER FastAPI
//
function buildNodePackages(startDir) {
    function walk(dir) {
        const files = readdirSync(dir);

        for (const file of files) {
            const full = path.join(dir, file);

            if (statSync(full).isDirectory()) {
                if (file !== "node_modules") walk(full);
            } else if (file === "package.json") {
                const pkg = JSON.parse(readFileSync(full, "utf8"));

                if (pkg.scripts?.build) {
                    console.log("Building:", dir);
                    execSync("npm run build", { cwd: dir, stdio: "inherit" });
                } else {
                    console.log("Skipping (no build script):", dir);
                }
            }
        }
    }

    walk(startDir);
}

//
// Execute in correct dependency order
//
const fastapiFolders = findFastAPIFolders(process.cwd());
console.log("Detected FastAPI folders:", fastapiFolders);

buildFastAPIBackends(fastapiFolders);
buildNodePackages(process.cwd());

