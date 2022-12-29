import { Routes, useLocation, useNavigate, useParams } from "react-router-dom";

const DemoOptionRoutes = (children) => {
  let location = useLocation();
  let state = location.state as { backgroundLocation?: Location };
  return (
    <>
      <Routes location={state?.backgroundLocation || location}>
        {children}
      </Routes>
    </>
  );
};

export default DemoOptionRoutes;
