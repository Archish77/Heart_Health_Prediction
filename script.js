function predict() {
    fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            age: age.value,
            sex: sex.value,
            cp: cp.value,
            trestbps: trestbps.value,
            chol: chol.value,
            fbs: fbs.value,
            restecg: restecg.value,
            thalach: thalach.value,
            exang: exang.value,
            oldpeak: oldpeak.value,
            slope: slope.value,
            ca: ca.value,
            thal: thal.value
        })
    })
    .then(res => res.json())
    .then(data => {
        result.innerHTML =
            `<b>${data.prediction}</b><br>Confidence: ${data.confidence}%`;
    });
}
