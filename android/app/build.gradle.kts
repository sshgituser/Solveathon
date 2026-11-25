import java.security.MessageDigest

plugins { alias(libs.plugins.android.application) }

val engineAar = providers.gradleProperty("qoldaEngineAar").orNull?.let { rootProject.file(it) }
val engineKit = providers.gradleProperty("qoldaEngineKit").orNull?.let { rootProject.file(it) }
check((engineAar == null) == (engineKit == null)) {
    "Supply both -PqoldaEngineAar and -PqoldaEngineKit, or neither for a source-check build."
}
val linked = engineAar != null
val approvedAarSha = "4e82863a8ea5e7c1ecedbdbce604e3e8f91b748113a33948459caebc462e4022"
val pinned = mapOf(
    "qolda/engine.json" to "754191c53b30b7880b2897b786a328643ceabeec38950e256e9225341a88bfa7",
    "qolda/a-tcn-fp32.tflite" to "99cb4409cad108ece6baddb3b1f90ad7c8bfc7e8906390dacd86fd29b6d7ec34",
    "qolda/rtmpose-fp32.tflite" to "2dd322a0c414e77fa591a82d0b670ec36c3e7547a57f42b121b9870b6ce5b8cd",
)

val gpuAssets = layout.buildDirectory.dir("generated/qolda/assets")
val webAssets = layout.buildDirectory.dir("generated/qolda/web-assets")
val frontend = rootProject.file("../frontend")
val stageWeb = tasks.register<Sync>("stageWebClient") {
    into(webAssets.map { it.dir("web") })
    from(frontend) {
        include("index.html")
        into("frontend")
        filter { line: String ->
            when {
                line.contains("http-equiv=\"Content-Security-Policy\"") ->
                    "<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'self'; connect-src 'self'; media-src 'self' blob:; img-src 'self' data: blob:; style-src 'self'; font-src 'self'; script-src 'self'; object-src 'none'; base-uri 'none'; frame-src 'none'; form-action 'none'\">"
                line.contains("name=\"seriq-profile\"") || line.contains("name=\"seriq-analyzer\"") || line.contains("name=\"seriq-api\"") -> ""
                line.trim() == "</head>" -> "<meta name=\"seriq-profile\" content=\"android\">\n<meta name=\"seriq-analyzer\" content=\"ondevice\">\n<script src=\"../android/bridge.js\"></script>\n</head>"
                else -> line
            }
        }
    }
    from(frontend) { include("src/**", "styles/**"); into("frontend") }
    from(rootProject.file("../assets")) {
        include("brand/**", "fonts/**", "models/mannequin.glb", "models/CREDITS.md", "vendor/three/**")
        into("assets")
    }
    from(file("src/main/bridge")) { include("bridge.js"); into("android") }
}
val stageGpuAssets = if (engineKit == null) null else tasks.register<Sync>("stageGpuAssets") {
    inputs.file(engineAar!!)
    inputs.property("approvedEngineAarSha256", approvedAarSha)
    from(engineKit.resolve("assets")) { include(pinned.keys); include("qolda/licenses/**") }
    into(gpuAssets)
    doFirst {
        check(engineAar!!.isFile && engineAar.length() == 4224419L &&
            MessageDigest.getInstance("SHA-256").digest(engineAar.readBytes()).joinToString("") { "%02x".format(it) } == approvedAarSha) {
            "T-018 AAR does not match the approved release."
        }
        for ((name, expected) in pinned) {
            val path = engineKit.resolve("assets/$name")
            check(path.isFile) { "Missing pinned asset $name" }
            val digest = MessageDigest.getInstance("SHA-256")
            path.inputStream().buffered().use { input ->
                val buf = ByteArray(65536)
                while (true) {
                    val count = input.read(buf)
                    if (count < 0) break
                    digest.update(buf, 0, count)
                }
            }
            check(digest.digest().joinToString("") { "%02x".format(it) } == expected) {
                "Pinned digest does not match $name"
            }
        }
    }
}

android {
    namespace = "net.qolda.offline"
    compileSdk = libs.versions.sdk.get().toInt()
    defaultConfig {
        applicationId = if (linked) "net.qolda.offline" else "net.qolda.offline.sourcecheck"
        minSdk = libs.versions.min.sdk.get().toInt()
        targetSdk = libs.versions.sdk.get().toInt()
        versionCode = 1
        versionName = if (linked) "0.1.0-t010" else "0.1.0-sourcecheck"
        buildConfigField("boolean", "ENGINE_LINKED", linked.toString())
        ndk { abiFilters += "arm64-v8a" }
    }
    buildFeatures { buildConfig = true }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    packaging {
        jniLibs {
            useLegacyPackaging = true
            // The verified runtime binaries must survive packaging byte-for-byte.
            keepDebugSymbols += "**/*.so"
        }
    }
    androidResources { noCompress += "tflite" }
    sourceSets.getByName("main").java.srcDir(if (linked) "src/gpu/java" else "src/sourcecheck/java")
    sourceSets.getByName("main").assets.srcDir(webAssets)
    if (linked) sourceSets.getByName("main").assets.srcDir(gpuAssets)
}

tasks.named("preBuild") { dependsOn(stageWeb) }
if (stageGpuAssets != null) {
    tasks.named("preBuild") { dependsOn(stageGpuAssets) }
    dependencies { implementation(files(engineAar!!)) }
}